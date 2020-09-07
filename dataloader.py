import json
import logging
import torch
from torch.nn.utils.rnn import  pad_sequence
from torch.utils.data import DataLoader
import random
from transformers import BertTokenizer
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s  %(message)s')

tag_idxs = {'B':0,'M':1,'E':2,'S':3,'O':4}

def collate_fn(batch):
    turn_mask = []
    nbatch = {}
    for b in batch:
        for i,t in enumerate(b):
            turn_mask.append(i)
            for k,v in t.items():
                nbatch[k]=nbatch.get(k,[])+[v]
    txt_ids = b['txt_ids']
    tags = b['tags']
    context_mask = b['context_mask']
    #下面进行padding操作
    blen = max([len(t) for t in txt_ids])
    ntxt_ids = pad_sequence(txt_ids,batch_first=True,padding_value=-1)
    ntags = pad_sequence(tags,batch_first=True,padding_value=-1)
    ncontext_mask = pad_sequence(context_mask,batch_first=True,padding_value=-1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[:txt_len]=1
    nbatch['txt_ids']=ntxt_ids
    nbatch['tags']=ntags
    nbatch['context_mask']=ncontext_mask
    nbatch['attention_mask']=attention_mask
    nbatch['turn_mask']=torch.tensor(turn_mask,dtype=torch.uint8)
    return nbatch

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

class MyDataset1:
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


    def batch_by_token(self):
        pass

    def batch_by_num(self):
        pass

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, i):
        return {'text_ids':self.text_ids[i],"token_type_ids":self.token_type_ids[i],"attention_mask":self.attention_mask[i],
                "target_tag":self.target_tag[i],"tuen_mask":self.turn_mask[i]}

def get_inputs(context,q,ans,tokenizer,max_len):
    """
    Args:
        context: 上下文句子
        q: 问题
        ans： 答案列表
        max_len: 允许的最大的长度
    Returns:
        txt_ids: [CLS]question[SEP]context[SEP]编码后的句子
        tags1: 对应的标注序列
        context_mask： 用来确定context的位置
    """
    context1 = [tokenizer.tokenize(c) for c in context.split()]
    context2 = tokenizer.tokenize(context)
    assert sum(context1,[])==context2
    query = tokenizer.tokenize(q)
    #将答案转化为tag的标注
    tags = [tag_idxs['O']]*len(context2)
    for i,an in enumerate(ans):
        if len(ans)==4:#第一轮对话的情况
            start,end,ent_str=an[1:]
        elif len(ans)==3:#提取尾实体的信息
            start,end,ent_str=an[-1][1:]
        s1 = [tokenizer.tokenize(t) for t in context[:start].split()]
        s2 = tokenizer.tokenize(context[start:end])
        assert sum(s1,[])==s2
        new_start = sum([len(s) for s in s1])
        new_end = new_start+len(s2)-1#这个end是取的闭区间
        if new_start!=new_end:
            tags[new_start]=tag_idxs['B']
            tags[new_end]=tag_idxs['E']
            tags[new_start+1:new_end]=tag_idxs['M']
        else:
            tags[new_start] = tag_idxs['S']
    txt_len = len(query)+len(context2)+3
    if txt_len>max_len:
        context2 = context2[:max_len-len(query)-3]
        tags = tags[:max_len-len(query)-3]
    txt = ['[CLS]']+query+['[SEP]']+context2+['[SEP]']
    txt_ids = tokenizer.convert_tokens_to_ids(txt)
    #[CLS]的tag用来判断是否存在答案
    tags1 = [tag_idxs['O'] if len(ans)>0 else -1]+[-1]*(len(query)+1)+tags+[-1]
    context_mask = [1]+[0]*(len(query)+1)+[0]*len(context2)+[0]
    return txt_ids,tags1,context_mask


class MyDataset:
    """
    这里假设我们的一个样本是两轮问答，或者一轮问答（第二轮为空）
    这可能导致第一轮问答被多个两轮问答作为首轮问答使用
    """
    def __init__(self,path,tokenizer,max_len=521):
        with open(path,encoding='utf-8') as f:
            data = json.load(f)
        self.all_qas = []
        for d in data:
            context = d['context']
            qa_pairs = d['qa_pairs']
            for t in qa_pairs:
                t1 = t[0]
                t2 = t[1]
                qas = []
                t1_qas = []
                t2_qas = []
                dict1 = {}#key:i,value:t1[i][ans],value是一个实体的列表
                dict2 = {}#key:t2[i][ans][0],value:i,
                for i,(q,ans) in enumerate(t1.items()):#这里的ans是某种类型的实体的列表
                    txt_ids,tags,context_mask = get_inputs(context,q,ans,tokenizer,max_len)
                    t1_qas.append({"txt_ids":txt_ids,"tags":tags,"context_mask":context_mask})
                    dict1[i] = ans
                for i,(q,ans) in enumerate(t2.items()):
                    txt_ids,tags,context_mask = get_inputs(context,q,ans,tokenizer,max_len)
                    t2_qas.append({"txt_ids":txt_ids,"tags":tags,"context_mask":context_mask})
                    key = (ans[0][-1][0],ans[0][-1][1],ans[0][-1][2])
                    dict2[key]=dict2.get(key,[])+[q]
                    dict2[ans[0][0]]=dict2.get(ans[0][0],[])+[i]
                #建立两轮问答的对关系
                for i,ans1 in dict1.items():
                    if len(ans1)>0:
                        for an in ans1:
                            assert dict2[an]!=0#所有的头实体都应该存在第二轮问答的
                            for j in dict2[an]:
                                qas.append([t1_qas[i],t2_qas[j]])
                    else:
                        #只有一轮问答的情况
                        qas.append([t1_qas[i]])
            self.all_qas.extend(qas)

    def __len__(self):
        #返回有多少个完整的问答（可能是一轮，也可能是两轮）
        return len(self.all_qas)

    def __getitem__(self, i):
        return self.all_qas[i]


class MyDownSampler:
    """对负样本样本进行下采样"""
    def __init__(self,dataset,ratio=1):
        """
        Args:
            ratio:正负样本的比例
        """
        self.ratio = ratio
        self.dataset = dataset
        #统计得到正/负样本的index
        self.positive_idxs = []
        self.negative_idxs = []
        for i in range(len(dataset)):
            d = dataset[i]
            if d[-1]['tags'][0]==-1:
                self.negative_idxs.append(i)
            else:
                self.positive_idxs.append(i)
        indexs = self.positive_idxs+random.sample(self.negative_idxs,len(self.positive_idxs)*self.ratio)
        indexs = random.sample(indexs,len(indexs))
        self.indexs = indexs
    def __iter__(self):
        return self.indexs
    def __len__(self):
        return len(self.indexs)


def load_data(file_path,batch_size,max_len,pretrained_model_path,down_ratio=1):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    dataset = MyDataset(file_path,tokenizer)
    sampler = MyDownSampler(dataset,down_ratio)
    dataloader = DataLoader(dataset,batch_size,sampler=sampler)
    return dataloader