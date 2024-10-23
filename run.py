
import logging

cpu_count = 12

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens_1,
                 input_ids_1,
                 position_idx_1,
                 dfg_to_code_1,
                 dfg_to_dfg_1,
                 input_tokens_2,
                 input_ids_2,
                 position_idx_2,
                 dfg_to_code_2,
                 dfg_to_dfg_2,
                 label,
                 url1,
                 url2
                 ):

        # The first code function
        self.input_tokens_1 = input_tokens_1
        self.input_ids_1 = input_ids_1
        self.position_idx_1 = position_idx_1
        self.dfg_to_code_1 = dfg_to_code_1
        self.dfg_to_dfg_1 = dfg_to_dfg_1

        # The second code function
        self.input_tokens_2 = input_tokens_2
        self.input_ids_2 = input_ids_2
        self.position_idx_2 = position_idx_2
        self.dfg_to_code_2 = dfg_to_code_2
        self.dfg_to_dfg_2 = dfg_to_dfg_2

        # label
        self.label = label
        self.url1 = url1
        self.url2 = url2

from .utils.code_to_dfg import (
    parsers,
    extract_data_flow_graph
)
from .utils.dfg_to_input_data import (
    dfg_to_graph_data
)

def convert_examples_to_features(item):
    # source
    url1, url2, label, tokenizer, args, cache, url_to_code = item
    parser = parsers['java']

    for url in [url1, url2]:
        if url not in cache:
            func = url_to_code[url]

            # extract data flow
            dfg = extract_data_flow_graph(func, parser, 'java')
            code_tokens = [tokenizer.tokenize(
                '@ '+x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
            ori2cur_pos = {}
            ori2cur_pos[-1] = (0, 0)
            for i in range(len(code_tokens)):
                ori2cur_pos[i] = (ori2cur_pos[i-1][1],
                                  ori2cur_pos[i-1][1]+len(code_tokens[i]))
            code_tokens = [y for x in code_tokens for y in x]

            # truncating
            code_tokens = code_tokens[:args.code_length+args.data_flow_length -
                                      3-min(len(dfg), args.data_flow_length)][:512-3]
            source_tokens = [tokenizer.cls_token] + \
                code_tokens+[tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            position_idx = [i+tokenizer.pad_token_id +
                            1 for i in range(len(source_tokens))]
            dfg = dfg[:args.code_length +
                      args.data_flow_length-len(source_tokens)]
            source_tokens += [x[0] for x in dfg]
            position_idx += [0 for x in dfg]
            source_ids += [tokenizer.unk_token_id for x in dfg]
            padding_length = args.code_length + \
                args.data_flow_length-len(source_ids)
            position_idx += [tokenizer.pad_token_id]*padding_length
            source_ids += [tokenizer.pad_token_id]*padding_length

            # reindex
            reverse_index = {}
            for idx, x in enumerate(dfg):
                reverse_index[x[1]] = idx
            for idx, x in enumerate(dfg):
                dfg[idx] = x[:-1]+([reverse_index[i]
                                   for i in x[-1] if i in reverse_index],)
            dfg_to_dfg = [x[-1] for x in dfg]
            dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
            length = len([tokenizer.cls_token])
            dfg_to_code = [(x[0]+length, x[1]+length) for x in dfg_to_code]
            cache[url] = source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg

    source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1 = cache[url1]
    source_tokens_2, source_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2 = cache[url2]
    return InputFeatures(source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1,
                         source_tokens_2, source_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2,
                         label, url1, url2)

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
    def __init__(self, tokenizer, args, file_path='train'):
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
                    data.append((url1, url2, label, tokenizer,
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
    pass

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