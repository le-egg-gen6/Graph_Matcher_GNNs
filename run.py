
import logging

cpu_count = 12

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 url1,
                 dfg1,
                 graph_data1,
                 url2,
                 dfg2,
                 graph_data2,
                 label
                 ):

        # The first code function
        self.dfg1 = dfg1
        self.graph_data1 = graph_data1

        # The second code function
        self.dfg2 = dfg2
        self.graph_data2 = graph_data2

        # label
        self.label = label
        self.url1 = url1
        self.url2 = url2

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

def convert_examples_to_features(item):
    # source
    url1, url2, label, args, cache, url_to_code = item
    parser = parsers['java']

    for url in [url1, url2]:
        if url not in cache:
            func = url_to_code[url]

            # extract data flow
            dfg = extract_data_flow_graph(func, parser, 'java')

            # convert to PyTorch Geometric Data
            graph_data = dfg_to_graph_data(dfg)

            cache[url] = dfg, graph_data

    dfg1, graph_data1 = cache[url1]
    dfg2, graph_data2 = cache[url2]
    return InputFeatures(url1, dfg1, graph_data1,
                         url2, dfg2, graph_data2,
                         label
                        )

import torch
import numpy as np
import json
import random
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt

torch.cuda.empty_cache()

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

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens_1: {}".format(
                    [x.replace('\u0120', '_') for x in example.input_tokens_1]))
                logger.info("input_ids_1: {}".format(
                    ' '.join(map(str, example.input_ids_1))))
                logger.info("position_idx_1: {}".format(
                    example.position_idx_1))
                logger.info("dfg_to_code_1: {}".format(
                    ' '.join(map(str, example.dfg_to_code_1))))
                logger.info("dfg_to_dfg_1: {}".format(
                    ' '.join(map(str, example.dfg_to_dfg_1))))

                logger.info("input_tokens_2: {}".format(
                    [x.replace('\u0120', '_') for x in example.input_tokens_2]))
                logger.info("input_ids_2: {}".format(
                    ' '.join(map(str, example.input_ids_2))))
                logger.info("position_idx_2: {}".format(
                    example.position_idx_2))
                logger.info("dfg_to_code_2: {}".format(
                    ' '.join(map(str, example.dfg_to_code_2))))
                logger.info("dfg_to_dfg_2: {}".format(
                    ' '.join(map(str, example.dfg_to_dfg_2))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask_1 = np.zeros((self.args.code_length+self.args.data_flow_length,
                                self.args.code_length+self.args.data_flow_length), dtype=bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx_1])
        max_length = sum([i != 1 for i in self.examples[item].position_idx_1])
        # sequence can attend to sequence
        attn_mask_1[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids_1):
            if i in [0, 2]:
                attn_mask_1[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code_1):
            if a < node_index and b < node_index:
                attn_mask_1[idx+node_index, a:b] = True
                attn_mask_1[a:b, idx+node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg_1):
            for a in nodes:
                if a+node_index < len(self.examples[item].position_idx_1):
                    attn_mask_1[idx+node_index, a+node_index] = True

        # calculate graph-guided masked function
        attn_mask_2 = np.zeros((self.args.code_length+self.args.data_flow_length,
                                self.args.code_length+self.args.data_flow_length), dtype=bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx_2])
        max_length = sum([i != 1 for i in self.examples[item].position_idx_2])
        # sequence can attend to sequence
        attn_mask_2[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids_2):
            if i in [0, 2]:
                attn_mask_2[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code_2):
            if a < node_index and b < node_index:
                attn_mask_2[idx+node_index, a:b] = True
                attn_mask_2[a:b, idx+node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg_2):
            for a in nodes:
                if a+node_index < len(self.examples[item].position_idx_2):
                    attn_mask_2[idx+node_index, a+node_index] = True

        return (torch.tensor(self.examples[item].input_ids_1),
                torch.tensor(self.examples[item].position_idx_1),
                torch.tensor(attn_mask_1),
                torch.tensor(self.examples[item].input_ids_2),
                torch.tensor(self.examples[item].position_idx_2),
                torch.tensor(attn_mask_2),
                torch.tensor(self.examples[item].label))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

##########-----Train_Stage
def train(args, train_dataset, matcher):
    
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)
    print(f"len( train_dataloader): {len( train_dataloader)}")
    args.max_steps = args.epochs*len(train_dataloader)
    args.save_steps = len(train_dataloader)//10
    args.warmup_steps = args.max_steps//5



    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    #Train
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",
                args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (inputs_ids_1, position_idx_1, attn_mask_1,
             inputs_ids_2, position_idx_2, attn_mask_2,
             labels) = [x.to(args.device) for x in batch]
            model.train()
            loss, logits = model(inputs_ids_1, position_idx_1, attn_mask_1,
                                 inputs_ids_2, position_idx_2, attn_mask_2, labels)

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer,
                                       eval_when_training=True)

                    # Save model checkpoint
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  "+"*"*20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  "+"*"*20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(
                            args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        output_dir = os.path.join(
                            output_dir, '{}'.format('model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info(
                            "Saving model checkpoint to %s", output_dir)


##########-----Evaluate_Stage
def evaluate(args, dataset, matcher):
    pass

##########-----Test_Stage
def test(args, dataset, matcher):
    pass

def main():
    pass

if __name__ == "__main__":
    main()