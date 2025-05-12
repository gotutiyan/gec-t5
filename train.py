
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler, SchedulerType
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from gec_t5.dataset import Dataset
from tqdm import tqdm
from collections import OrderedDict
import json
from accelerate import Accelerator
import numpy as np
import random
import datetime

def train(
    model,
    loader: DataLoader,
    optimizer,
    epoch: int,
    accelerator: Accelerator,
    lr_scheduler: SchedulerType,
) -> float:
    model.train()
    log = {
        'loss': 0
    }
    with tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_main_process) as pbar:
        for _, batch in pbar:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                log['loss'] += loss.item()
                if accelerator.is_main_process:
                    pbar.set_description(f'[Epoch {epoch}] [TRAIN]')
                    pbar.set_postfix(OrderedDict(
                        loss=loss.item(),
                        lr=optimizer.optimizer.param_groups[0]['lr']
                    ))
    return {k: v/len(loader) for k, v in log.items()}

def valid(model,
    loader: DataLoader,
    epoch: int,
    accelerator: Accelerator
) -> float:
    model.eval()
    log = {
        'loss': 0
    }
    with torch.no_grad():
        with tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_main_process) as pbar:
            for _, batch in pbar:
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    log['loss'] += loss.item()
                    if accelerator.is_main_process:
                        pbar.set_description(f'[Epoch {epoch}] [VALID]')
                        pbar.set_postfix(OrderedDict(
                            loss=loss.item()
                        ))
    return {k: v/len(loader) for k, v in log.items()}

def readlines(path):
    lines = open(path).readlines()
    lines = [l.rstrip() for l in lines]
    print(lines[:5])
    return lines

def main(args):
    config = json.load(open(os.path.join(args.restore_dir, 'my_config.json'))) if args.restore_dir else dict()
    model_id = config.get('model_id', args.model_id)
    current_epoch = config.get('epoch', 0)
    min_valid_loss = config.get('min_valid_loss', float('inf'))
    seed = config.get('seed', args.seed)
    log_dict = json.load(open(os.path.join(args.restore_dir, '../log.json'))) if args.restore_dir else dict()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print('Loading model...')
    model = AutoModelForSeq2SeqLM.from_pretrained(args.restore_dir) if args.restore_dir else AutoModelForSeq2SeqLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.restore_dir) if args.restore_dir else AutoTokenizer.from_pretrained(model_id)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    print('Loading datasets...')
    train_dataset = Dataset(
        srcs=readlines(args.trainpref+'.'+args.source),
        trgs=readlines(args.trainpref+'.'+args.target),
        tokenizer=tokenizer,
        max_len=args.max_len
    )
    valid_dataset = Dataset(
        srcs=readlines(args.validpref+'.'+args.source),
        trgs=readlines(args.validpref+'.'+args.target),
        tokenizer=tokenizer,
        max_len=args.max_len
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    os.makedirs(os.path.join(args.outdir, 'best'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'last'), exist_ok=True)
    tokenizer.save_pretrained(os.path.join(args.outdir, 'best'))
    tokenizer.save_pretrained(os.path.join(args.outdir, 'last'))
    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_loader) * args.epochs,
    )
    if args.restore_dir:
        model.load_state_dict(torch.load(os.path.join(dir, 'lr_state.bin')))
    model, optimizer, train_loader, valid_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, lr_scheduler
    )
    print('Start training ...')
    for epoch in range(current_epoch, args.epochs):
        train_log = train(model, train_loader, optimizer, epoch, accelerator, lr_scheduler)
        valid_log = valid(model, valid_loader, epoch, accelerator)
        log_dict[f'Epoch {epoch}'] = {
            'train_log': train_log,
            'valid_log': valid_log,
            'time': str(datetime.datetime.now())
        }
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if min_valid_loss > valid_log['loss']:
                torch.save(lr_scheduler.state_dict(), os.path.join(args.outdir, 'best/lr_state.bin'))
                accelerator.unwrap_model(model).save_pretrained(os.path.join(args.outdir, 'best'))
                min_valid_loss = valid_log['loss']
                config_dict = {
                    'model_id': model_id,
                    'epoch': epoch,
                    'min_valid_loss': min_valid_loss,
                    'seed': args.seed,
                    'argparse': args.__dict__
                }
                with open(os.path.join(args.outdir, 'best/my_config.json'), 'w') as fp:
                    json.dump(config_dict, fp, indent=4)
            with open(os.path.join(args.outdir, 'log.json'), 'w') as fp:
                json.dump(log_dict, fp, indent=4)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(lr_scheduler.state_dict(), os.path.join(args.outdir, 'last/lr_state.bin'))
        accelerator.unwrap_model(model).save_pretrained(os.path.join(args.outdir, 'last'))
        config_dict = {
            'model_id': model_id,
            'epoch': epoch,
            'min_valid_loss': min_valid_loss,
            'seed': args.seed,
            'argparse': args.__dict__
        }
        with open(os.path.join(args.outdir, 'last/my_config.json'), 'w') as fp:
            json.dump(config_dict, fp, indent=4)
        print('Finish')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainpref', required=True)
    parser.add_argument('--validpref', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--model_id', default='google/t5-v1_1-base')
    parser.add_argument('--outdir', default='models/sample/')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--restore_dir', default=None)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)
