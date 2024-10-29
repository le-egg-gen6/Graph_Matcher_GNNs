
from .utils.code_to_dfg import (
    parsers,
    extract_data_flow_graph
)
from .utils.dfg_to_input_data import (
    dfg_to_graph_data,
    batch_dfg_to_pyg,
    create_graph_matcher_input
)

from .utils.tokenizer_utils import (
    initialize_pretrained_tokenizer_and_model,
    get_code_token_embeddings
)

from .model.graph_matcher_model import (
    CodeSimilarityDetectionModel
)


import os
import torch
import numpy as np
import json
import random
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import torch.nn.functional as F
from tqdm import tqdm
import multiprocessing
import argparse

import logging

cpu_count = multiprocessing.cpu_count()

logger = logging.getLogger(__name__)

torch.cuda.empty_cache()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 url1, dfg1, code_token1,
                 url2, dfg2, code_token2,
                 label
                 ):

        # The first code function
        self.dfg1 = dfg1
        self.code_token1 = code_token1

        # The second code function
        self.dfg2 = dfg2
        self.code_token2 = code_token2

        # label
        self.label = label
        self.url1 = url1
        self.url2 = url2

def convert_examples_to_features(item, tokenizer=None):
    # source
    url1, url2, label, args, cache, url_to_code = item
    parser = parsers['java']

    for url in [url1, url2]:
        if url not in cache:
            func = url_to_code[url]

            # extract data flow and code token from src code
            dfg, code_tokens = extract_data_flow_graph(func, parser, 'java')

            cache[url] = dfg, code_tokens

    dfg1, code_token1 = cache[url1]
    dfg2, code_token2 = cache[url2]
    return InputFeatures(url1, dfg1, code_token1,
                         url2, dfg2, code_token2,
                         label)

class TextDataset(Dataset):
    def __init__(self, args, file_path='dataset'):
        self.examples = []
        self.args = args
        index_filename = file_path

        # load index
        logger.info("Creating features from index file at %s ", index_filename)
        url_to_code = {}
        with open('/'.join(index_filename.split('/')[:-1])+'/data.jsonl') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                url_to_code[js['idx']] = js['func']

        # load code function according to index
        data = []
        cache = {}
        f = open(index_filename)
        with open(index_filename) as f:
            for line in f:
                line = line.strip()
                url1, url2, label = line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                # Convert label to integer
                label = int(label)
                # Check if label is one of the four classes
                if label in [0, 1, 2, 3]:
                    data.append((url1, url2, label,
                                args, cache, url_to_code))
                else:
                    logger.error("Invalid label: %s", label)

        # only use 10% valid data to keep best model
        if 'valid' in file_path:
            data = random.sample(data, int(len(data)*0.1))

        # convert example to input features
        self.examples = [convert_examples_to_features(
            x) for x in tqdm(data, total=len(data))]

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return self.examples[index]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr after
    a warmup period during which it increases linearly from 0 to the initial lr.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: The number of steps for the warmup phase
        num_training_steps: The total number of training steps
        last_epoch: The index of the last epoch when resuming training
    
    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule
    """
    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Linear decay phase
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


##########-----Train_Stage
def train(args, train_dataset, model, tokenizer=None, model_tokenizer=None):
    #Move model to CPU if necessary
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)
    
    """Train the model"""
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=args.train_batch_size,
        num_workers=cpu_count//2,
        pin_memory=True
    )
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                              num_training_steps=t_total)


    # Initialize loss tracking
    losses = []  # Track all losses
    epoch_losses = []  # Track losses per epoch
    running_loss = 0.0  # Running loss for current epoch
    best_avg_loss = float('inf')
    
    # Learning rate history
    lr_history = []
    
    # Training metrics logger
    metrics = {
        'epoch': [],
        'step': [],
        'loss': [],
        'learning_rate': [],
        'avg_epoch_loss': []
    }

    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    
    train_iterator = tqdm(range(int(args.epochs)), desc="Epoch")
    
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Training")
        epoch_loss = 0.0
        num_batches = 0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            try:
                dfgs_1 = [example.dfg1 for example in batch]
                code_tokens_1 = [example.code_token1 for example in batch]
                code_token_embeddeds_1 = None

                dfgs_2 = [example.dfg2 for example in batch]
                code_tokens_2 = [example.code_token2 for example in batch]
                code_token_embeddeds_2 = None

                if tokenizer is not None and model_tokenizer is not None:
                    code_token_embeddeds_1 = get_code_token_embeddings(code_tokens_1, tokenizer, model_tokenizer)
                    code_token_embeddeds_2 = get_code_token_embeddings(code_tokens_2, tokenizer, model_tokenizer)
                # Prepare batch data
                batch_data_1 = batch_dfg_to_pyg(dfgs_1, code_tokens_1, code_token_embeddeds_1)
                batch_data_2 = batch_dfg_to_pyg(dfgs_2, code_tokens_2, code_token_embeddeds_2)
                
                labels = torch.tensor([example.label for example in batch]).to(args.device)
                
                # Move data to GPU if available
                batch_data_1 = batch_data_1.to(args.device)
                batch_data_2 = batch_data_2.to(args.device)
                
                # Forward pass
                outputs = model(batch_data_1, batch_data_2)
                loss = F.cross_entropy(outputs, labels)

                # Backward pass
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                # Track losses
                batch_loss = loss.item()
                tr_loss += batch_loss
                epoch_loss += batch_loss
                losses.append(batch_loss)
                num_batches += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # Optimizer and scheduler steps
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    # Track learning rate
                    current_lr = scheduler.get_last_lr()[0]
                    lr_history.append(current_lr)
                    
                    # Log metrics
                    metrics['step'].append(global_step)
                    metrics['loss'].append(batch_loss)
                    metrics['learning_rate'].append(current_lr)
                    
                    # Update progress bar
                    epoch_iterator.set_postfix({
                        'loss': f'{batch_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'avg_loss': f'{tr_loss/global_step:.4f}'
                    })

                    # Log every N steps
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logger.info(
                            f'Step {global_step}: '
                            f'Loss = {tr_loss/global_step:.4f}, '
                            f'LR = {current_lr:.2e}'
                        )

                    # Evaluation and checkpoint saving
                    if args.evaluate_during_training and (global_step % args.eval_steps == 0):
                        results = evaluate(args, model, tokenizer)
                        
                        output_dir = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        
                        # Save training metrics
                        torch.save({
                            'losses': losses,
                            'lr_history': lr_history,
                            'metrics': metrics
                        }, os.path.join(output_dir, 'training_metrics.pt'))
                        
                        logger.info(f"Saving model checkpoint to {output_dir}")


            except Exception as e:
                logger.error(f"Error in training step {step}: {str(e)}")
                continue

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        
        # End of epoch processing
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        metrics['epoch'].append(epoch)
        metrics['avg_epoch_loss'].append(avg_epoch_loss)
        
        logger.info(f"Epoch {epoch} completed: Average Loss = {avg_epoch_loss:.4f}")
        
        # Save best model based on average epoch loss
        if avg_epoch_loss < best_avg_loss:
            best_avg_loss = avg_epoch_loss
            output_dir = os.path.join(args.output_dir, 'best_model')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))
            logger.info(f"New best model saved with average loss: {best_avg_loss:.4f}")


        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # Save final training metrics
    final_metrics = {
        'final_loss': tr_loss / global_step,
        'best_loss': best_avg_loss,
        'losses': losses,
        'epoch_losses': epoch_losses,
        'lr_history': lr_history,
        'metrics': metrics
    }
    
    torch.save(final_metrics, os.path.join(args.output_dir, 'final_training_metrics.pt'))

    return global_step, tr_loss / global_step


##########-----Evaluate_Stage
def evaluate(args, train_dataset, model, tokenizer=None, model_tokenizer=None, eval_during_training=False):
    pass

##########-----Test_Stage
def test(args, train_dataset, model, tokenizer=None, model_tokenizer=None, thresh_hold=0):
    pass

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")

    # Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    

    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Logging Metric After Pre-Defined Steps")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument('out')

    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu,)

    # Set seed
    set_seed(args)
    tokenizer, model = initialize_pretrained_tokenizer_and_model(args.tokenizer_name) if args.tokenizer_name is not "" else initialize_pretrained_tokenizer_and_model()

    matcher_model =  #Need Complete
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = TextDataset(
            tokenizer, args, file_path=args.train_data_file)
        train(args, train_dataset, matcher_model, tokenizer)

if __name__ == "__main__":
    main()