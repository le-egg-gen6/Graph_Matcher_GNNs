from model import (
    CodeSimilarityDetectionModel,
    CloneDetectionLoss
)

from parsers import (
    DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
)

import os
import torch
import numpy as np
import json
import random
from torch.utils.data import DataLoader, Dataset, RandomSampler
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

class InputFeature:
    def __init__(self, code1, code2, label):
        self.code1 = code1
        self.code2 = code2
        self.label = label

class TextDataset(Dataset):
    def __init__(self, args, file_path='dataset'):
        self.examples = []
        self.args = args
        
        logger.info("Creating features from index file at %s", file_path)
        
        # Load code data
        data_file = os.path.join(os.path.dirname(file_path), 'data.jsonl')
        url_to_code = {}
        
        try:
            with open(data_file) as f:
                for line in f:
                    try:
                        line = line.strip()
                        js = json.loads(line)
                        url_to_code[js['idx']] = js['func']
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line: {line}")
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_file}")
            
        logger.info(f"Loaded {len(url_to_code)} code snippets")
        
        # Load and process pairs
        try:
            with open(file_path) as f:
                for line in f:
                    try:
                        line = line.strip()
                        url1, url2, label = line.split('\t')
                        
                        # Skip if either code is missing
                        if url1 not in url_to_code or url2 not in url_to_code:
                            continue
                            
                        # Convert label to integer
                        label = int(label)
                        
                        # Check if label is valid
                        if label in [0, 1, 2, 3]:
                            example = InputFeature(
                                code1=url_to_code[url1],
                                code2=url_to_code[url2],
                                label=label
                            )
                            self.examples.append(example)
                        else:
                            logger.warning(f"Skipping invalid label: {label}")
                            
                    except ValueError:
                        logger.warning(f"Skipping malformed line: {line}")
                        continue
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"Index file not found: {file_path}")
            
        # Validate dataset size
        if len(self.examples) == 0:
            raise ValueError(
                "No valid examples found in dataset. "
                "Please check your data files and make sure they contain valid examples."
            )
            
        logger.info(f"Loaded {len(self.examples)} valid example pairs")
        
        # Subsample validation set if needed
        if 'valid' in file_path:
            sample_size = int(len(self.examples) * 0.1)
            self.examples = random.sample(self.examples, sample_size)
            logger.info(f"Subsampled to {len(self.examples)} validation examples")

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
def collate_fn(batch):
    """
    Custom collate function for batching InputFeatures objects
    Args:
        batch: List of InputFeatures objects
    Returns:
        Dictionary containing batched tensors
    """
    # Lấy all code tokens và labels từ batch
    code1_s = [item.code1 for item in batch]
    code2_s = [item.code2 for item in batch]
    labels = torch.tensor([item.label for item in batch], dtype=torch.long)
    
    return {
        'code1_s': code1_s,
        'code2_s': code2_s,
        'labels': labels
    }


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
def train(args, train_dataset, model):
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)
    
    # Initialize custom loss function with class weights if specified
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(args.device)  # Can be adjusted based on class distribution
    criterion = CloneDetectionLoss(weights=class_weights)
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=args.train_batch_size,
        num_workers=cpu_count//2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Calculate total training steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs

    # Prepare optimizer and schedule
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

    # Training metrics tracking
    metrics = {
        'epoch': [],
        'step': [],
        'loss': [],
        'learning_rate': [],
        'avg_epoch_loss': []
    }
    
    global_step = 0
    tr_loss = 0.0
    best_avg_loss = float('inf')
    model.zero_grad()
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Batch size per device = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    train_iterator = tqdm(range(int(args.epochs)), desc="Epoch")
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)
    
    # Initialize custom loss function with class weights if specified
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(args.device)
    criterion = CloneDetectionLoss(weights=class_weights)
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_sampler = RandomSampler(train_dataset)
    
    # Sử dụng collate_fn tùy chỉnh trong DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=args.train_batch_size,
        num_workers=cpu_count//2,
        pin_memory=True,
        collate_fn=collate_fn  # Thêm collate_fn vào đây
    )
    
    # Calculate total training steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs

    # Prepare optimizer and schedule
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
    
    # Training loop
    global_step = 0
    tr_loss = 0.0
    model.zero_grad()

    # isLogged = False
    
    train_iterator = tqdm(range(int(args.epochs)), desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            try:
                # Process code pairs
                code1_s = batch['code1_s']
                code2_s = batch['code2_s']
                labels = batch['labels'].to(args.device)

                # if not isLogged:
                #     logger.info(str(code1_s[0]))
                #     logger.info(str(code2_s[0]))
                #     logger.info(labels[0])
                #     isLogged = True
                
                # Forward pass
                outputs = model(code1_s, code2_s)
                loss = criterion(outputs, labels)
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                tr_loss += loss.item()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1
                    
                    epoch_iterator.set_description(f"Loss: {loss.item():.4f}")
                    
                    if args.evaluate_during_training and (global_step % args.eval_steps == 0):
                        # Save checkpoint
                        output_dir = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))
                        
            except Exception as e:
                logger.error(f"Error in training step {step}: {str(e)}")
                continue
                
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
                
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    
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
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory for model checkpoints.")
    
    # Model parameters
    parser.add_argument("--model_name", default="microsoft/codebert-base", type=str,
                        help="The pre-trained model name for code embeddings")
    parser.add_argument("--input_dim", type=int, default=768,
                        help="Input dimension for embeddings")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for the model")
    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Output embedding dimension")
    
    # Training parameters
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run evaluation.")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before backward pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay for optimization.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup steps.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    
    # Evaluation parameters
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to evaluate during training.")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Evaluate every X steps during training.")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="If > 0: set total number of training steps. Override num_train_epochs.")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    logger.info(f"Device: {device}, Number of GPUs: {args.n_gpu}")
    
    # Set seed
    set_seed(args)
    
    # Initialize model
    dfg_functions = {
        "python" : DFG_python,
        "java" : DFG_java,
        "ruby" : DFG_ruby,
        "go" : DFG_go,
        "php" : DFG_php,
        "javascript" : DFG_javascript
    } 
    model = CodeSimilarityDetectionModel(
        dfg_functions=dfg_functions,
        model_name=args.model_name,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim
    )
    
    # Training
    if args.do_train:
        train_dataset = TextDataset(args, file_path=args.train_data_file)
        global_step, tr_loss = train(args, train_dataset, model)
        logger.info(f" Training completed: {global_step} steps, Average loss = {tr_loss}")
        
        # Save final model
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))
        logger.info(f"Model saved to {args.output_dir}")
    
    # Evaluation
    if args.do_eval:
        logger.info("Evaluation not implemented yet")

if __name__ == "__main__":
    main()