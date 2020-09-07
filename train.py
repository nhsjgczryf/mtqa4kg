import tqdm
import time
import random
import argparse
import numpy as np
import torch
from model import MyModel
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s  %(message)s')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path")
    parser.add_argument("--dev_path")
    parser.add_argument("--train_batch")
    parser.add_argument("--dev_batch")
    parser.add_argument("--pretrained_model_path")
    parser.add_argument("--max_epochs")
    parser.add_argument("--warmup_ratio",type=float)
    parser.add_argument("--lr",type=float)
    parser.add_argument("--dropout_prob",type=float)
    parser.add_argument("--weight_decay",type=float)
    parser.add_argument("--theta",type=float,help="调节两个任务的权重")
    parser.add_argument("--debug",action="store_true")
    parser.add_argument("--local_rank",type=int,default=-1)
    parser.add_argument("--max_gad_norm",type=float)
    parser.add_argument("--seed",type=int,default=209)
    args = parser.parse_args()
    return args

def train(args,train_dataloader,dev_dataloader=None):
    model = MyModel(args)
    model.train()
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay":args.weight_decay},
        {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay":0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr)
    if args.warmup_ratio>0:
        num_training_steps = len(train_dataloader)*args.max_epochs
        warmup_steps = args.warmup_ratio*num_training_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
        for epoch in range(args.max_epochs):
             

if __name__=="__main__":
    args = args_parser()
    set_seed(args.seed)