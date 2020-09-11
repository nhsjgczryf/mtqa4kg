from tqdm import trange,tqdm
import time
import random
import argparse
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm
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
    parser.add_argument("--max_gad_norm",type=float,default=1)
    parser.add_argument("--seed",type=int,default=209)
    args = parser.parse_args()
    return args

def train(args,train_dataloader,dev_dataloader=None):
    model = MyModel(args)
    model.train()
    device = torch.device('cuda:%d'%args.local_rank) if args.local_rank!=-1 else torch.device('cpu')
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
    log_dir = "logs/scalars/"+time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime(time.time()))
    writer = SummaryWriter(log_dir)
    for epoch in trange(args.max_epochs):
        for i,batch in tqdm(enumerate(train_dataloader)):
            txt_ids, attention_mask, token_type_ids, context_mask, turn_mask=batch['txt_ids'],batch['attention_mask'],\
                                                                           batch['token_type_ids'],batch['context_mask'],batch['turn_mask']
            txt_ids, attention_mask, token_type_ids, context_mask, turn_mask = txt_ids.to(device),attention_mask.to(device),\
                                                                               token_type_ids.to(device),context_mask.to(device),turn_mask.to(device)
            loss,(loss_t1,loss_t2) = model(txt_ids, attention_mask, token_type_ids, context_mask, turn_mask)
            loss.backward()
            if args.max_gad_norm>0:
                clip_grad_norm(model.parameters(),args.max_gad_norm)
            writer.add_histogram('gradient',model.parameters().grad,epoch*len(train_dataloader)+i)
            optimizer.step()
            if args.warmup_ratio>0:
                scheduler.step()
            writer.add_scalars('loss',{'overall':loss.item(),'turn1':loss_t1,'turn2':loss_t2},epoch*len(train_dataloader)+i)
    if args.local_rank in [-1,0]:
        if hasattr(model,'module'):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        scheduler_state_dict = optimizer.state_dict()
        checkpoint = {"model_state_dict":model_state_dict,"optimizer_state_dict":optimizer_state_dict,
                      "scheduler_state_dict":scheduler_state_dict}
        torch.save(checkpoint)

if __name__=="__main__":
    args = args_parser()
    set_seed(args.seed)