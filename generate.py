
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from dataset import generate_dataset
from tqdm import tqdm
from collections import OrderedDict
import json
from accelerate import Accelerator
import numpy as np
import random

def generate(
    model,
    loader: DataLoader,
    n_dataset: int
) -> float:
    model.eval()
    pred_ids = [0] * n_dataset
    with torch.no_grad():
        with tqdm(enumerate(loader), total=len(loader)) as pbar:
            for _, batch in pbar:
                batch = {k:v.cuda() for k,v in batch.items()}
                ids = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=128,
                    num_beams=5,
                    do_sample=False,
                    length_penalty=1.0
                )
                orig_index = batch['orig_index'].tolist()
                for i, p_id in enumerate(ids):
                    pred_ids[orig_index[i]] = p_id
    return pred_ids

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    model = AutoModelForSeq2SeqLM.from_pretrained(args.restore_dir)
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.restore_dir)
    dataset = generate_dataset(
        src_file=args.input,
        trg_file=args.input,
        tokenizer=tokenizer,
        max_len=args.max_len,
        is_inference=True
    )
    dataset.sort_by_length()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    pred_ids = generate(model, loader, n_dataset=len(dataset))
    predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print('\n'.join(predictions))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--model_id', default='facebook/bart-base')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--restore_dir', required=True)
    parser.add_argument('--max_len', type=int, default=128)


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)
