import torch
import torch.nn as nn
import logging
from transformers import BertModel
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s  %(message)s')

class MyModel(nn.Module):
    """
    这个模型时一轮问答
    """
    def __init__(self, config):
        self.config = config
        self.bert = BertModel.from_pretrained(config.pretrained_model_name_or_path)
        self.tag_linear = nn.Linear(config.hidden_size, 4)#BMEO标注，4种标签
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, input_ids, attention_mask, token_type_ids, target_tag,turn_mask):
        """
        Desc:
            一轮问答
        Args:
            input_ids: (batch_size,turn_num,seq_len)
            target_tag: (batch_size,turn_num,seq_len)
            turn_mask: BoolTensor (batch_size,turn_num) turn_num[i][j]代表第i个样本里面的第j轮对话是否存在
        Return:
            context中每个token的tag概率分布
        """
        rep,_cls = self.bert(input_ids,attention_mask,token_type_ids)
        if not target_mask is None:
            context_rep = rep[target_mask] #(N, hidden_size)，N是batch中context token的数量
            context_target_tags = target_tag[target_mask] #(N, hidden_size)
            context_rep_logit = self.tag_linear(self.dropout(context_rep)) #(N)
            loss = nn.functional.cross_entropy(context_target_tags,context_rep_logit)
            return loss
        else:
            rep = self.tag_linear(self.dropout(rep)) #(batch,seq_len,5)
        return rep

class MyModel(nn.Module):
    def __init__(self,config):
        self.config=config
        self.bert = BertModel.from_pretrained(config.pretrained_model_name_or_path)
        self.tag_linear = nn.Linear(self.bert.config.hidden_size,4)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.loss_func = nn.CrossEntropyLoss()
        self.theta = config.theta

    def forward(self,input,attention_mask,token_type_ids,context_mask=None,turn_mask=None,target_tags=None):
        """
        Args:
            input: （batch,seq_len），batch里面可能有第一轮的问答，也可能有第二轮的问答
            attention_mask:(batch,seq_len)
            token_type_ids:(batch,seq_len)
            context_mask:(batch,seq_len)，context用来确认拥有标注的token，注意为了处理无答案的情况[CLS]也属于context
            target_tags:(batch,seq_len)
            turn_mask:(batch,) turn_mask[i]=0代表第一轮，turn_mask[i]=1代表第2轮
        """
        rep,cls_rep = self.bert(input,attention_mask,token_type_ids)
        rep = self.dropout(rep)
        tag_logits = self.tag_linear(rep) #(batch,seq_len,4)
        if target_tags is None:
            #训练的情形
            tag_logits_t1 = tag_logits[turn_mask==0]#(n1,seq_len,4)
            target_tags_t1 = target_tags[turn_mask==0]#(n1,seq_len)
            context_mask_t1 = context_mask[turn_mask==0]#(n1,seq_len)
            tag_logits_t2 = tag_logits[turn_mask==1]#(n2,seq_len,4)
            target_tags_t2 = tag_logits[turn_mask==1]#(n2,seq_len)
            context_mask_t2 = context_mask[turn_mask==1]#(n2,seq_len)
            tag_logits_t1 = tag_logits_t1[context_mask_t1==1]#(N1,4)
            target_tags_t1 = target_tags_t1[context_mask_t1==1]#(N1.4)
            tag_logits_t2 = tag_logits_t2[context_mask_t2==1]#(N2,4)
            target_tags_t1 = target_tags_t1[context_mask_t2==1]#(N2)
            loss_t1 = self.loss_func(tag_logits_t1,target_tags_t1)
            loss_t2 = self.loss_func(tag_logits_t2,target_tags_t2)
            loss = self.theta*loss_t1+(1-self.theta)*loss_t2
            return loss,(loss_t1.item(),loss_t2.item())#后面一项主要用于训练的时候进行记录子任务的损失
        else:
            #预测的情形
            tag_idxs = torch.argmax(tag_logits,dim=-1).squeeze(-1)#(batch,seq_len)
            return tag_idxs