import json
import logging
import torch
from torch.nn.utils.rnn import  pad_sequence
from torch.utils.data import DataLoader
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s  %(message)s')


def collate_fn(batch):
    pass

def trans(tokenizer,context,tags):
    """
    将原始的query和context进行subword tokenize处理
    :param tags: 是一轮对话的所有tag，所以是一个双重列表
    """
    #context用空白作为单词的分隔符
    context = context.split()
    context1 = []
    tags1 = []
    for c in context:
        context1.append(tokenizer.tokenize(c))
    for i in range(len(tags)):
        tag = []
        for j,cs in enumerate(context1):
            tag.extend([tags[i][j]]*len(cs))
        tags1.append(tag)
    context2 = []
    for c in context1:
        context2.extend(c)
    return context2,tags1

class MyDataset:
    '''
    这个类支持两种构造python
    '''
    def __init__(self,path,tokenizer,max_turn,max_len=512,batch_size=-1,max_tokens=-1):
        """
        Desc:
            支持两种构造数据集的方式，
            第一种是不考虑样本长度不平衡的现象，直接进行batch采样
            第二种是以max_token的方式
        Args:
            path: 文件路径，文件中的内容为
            max_turn: 对话的轮数
            max_tokens: 一个batch里所包含的最多的token的数量
            max_len: 允许的最大输入长度
        """
        self.text_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.target_tag = []
        self.turn_mask = []
        with open(path,encoding='utf-8') as f:
            self.data = json.load(f)
        for d in self.batch:
            context = d['context']
            querys = d['querys']#querys是一轮对话的k个query
            tags = d['tags']#context对应的标注序列
            turn_num = len(querys)
            querys1 = [tokenizer.tokenize(query.split()) for query in querys]
            context1, tags1 = trans(tokenizer,context,tags)
            texts = []
            for i in range(turn_num):
                text = ['CLS']+querys1[i]+['SEP']+context1+['SEP']
                text = tokenizer.convert_tokens_to_ids(text)
                texts.append(text)
            texts = texts + [[]*(max_turn-turn_num)]
            max_seq_len = max([len(t) for t in texts])
            text_ids = []
            for i in range(max_turn):
                text_id =
                torch.full()

    def batch_by_token(self):
        pass

    def batch_by_num(selfs):
        pass

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, i):
        return {'text_ids':self.text_ids[i],"token_type_ids":self.token_type_ids[i],"attention_mask":self.attention_mask[i]
                "target_tag":self.target_tag[i],"tuen_mask":self.turn_mask[i]}


class BatchDataet:
    """
    这个类用于DistributedDataParallel训练，主要是把一个batch封装为Dataset里的一个Item
    """
    def __init__(self,path, max_tokens):
        pass
    def __len__(self):
        pass
    def __getitem__(self, i):
        pass