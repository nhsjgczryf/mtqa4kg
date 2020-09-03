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
        self.tag_linear = nn.Linear(config.hidden_size, 5)#BMESO标注，5种标签
        self.dropout = nn.Droupout(config.dropout_prob)

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
        ""
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
