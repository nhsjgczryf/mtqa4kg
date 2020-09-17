import json
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import  pad_sequence
from torch.utils.data import DataLoader,DistributedSampler
import random
from transformers import BertTokenizer
import preprocess

tag_idxs = {'B':0,'M':1,'E':2,'S':3,'O':4}

def collate_fn(batch):
    turn_mask = []
    nbatch = {}
    for b in batch:
        for i,t in enumerate(b):
            turn_mask.append(i)
            for k,v in t.items():
                nbatch[k]=nbatch.get(k,[])+[torch.tensor(v)]
    txt_ids = nbatch['txt_ids']
    tags = nbatch['tags']
    context_mask = nbatch['context_mask']
    token_type_ids = nbatch['token_type_ids']
    #下面进行padding操作
    ntxt_ids = pad_sequence(txt_ids,batch_first=True,padding_value=0)#padding的值要在词汇表里面
    ntags = pad_sequence(tags,batch_first=True,padding_value=-1)
    ncontext_mask = pad_sequence(context_mask,batch_first=True,padding_value=0)#这个和之前对query的padding一致
    ntoken_type_ids = pad_sequence(token_type_ids,batch_first=True,padding_value=1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[:txt_len]=1
    nbatch['txt_ids']=ntxt_ids
    nbatch['tags']=ntags
    nbatch['context_mask']=ncontext_mask
    nbatch['token_type_ids']=ntoken_type_ids
    nbatch['attention_mask']=attention_mask
    nbatch['turn_mask']=torch.tensor(turn_mask,dtype=torch.uint8)
    return nbatch

def collate_fn_t1(batch):
    nbatch = {}
    for b in batch:
        for k,v in b.items():
            nbatch[k]=nbatch.get(k,[])+[torch.tensor(v)]
    txt_ids = nbatch['txt_ids']
    context_mask = nbatch['context_mask']
    token_type_ids = nbatch['token_type_ids']
    ntxt_ids = pad_sequence(txt_ids,batch_first=True,padding_value=0)#padding的值要在词汇表里面
    ncontext_mask = pad_sequence(context_mask,batch_first=True,padding_value=0)#这个和之前对query的padding一致
    ntoken_type_ids = pad_sequence(token_type_ids,batch_first=True,padding_value=1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[:txt_len]=1
    nbatch['txt_ids']=ntxt_ids
    nbatch['context_mask']=ncontext_mask
    nbatch['token_type_ids']=ntoken_type_ids
    nbatch['attention_mask']=attention_mask
    return nbatch
    

def get_inputs(context,q,tokenizer,max_len=200,ans=[]):
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
    #context1 = [tokenizer.tokenize(c) for c in context.split()]
    context2 = tokenizer.tokenize(context)
    #assert sum(context1,[])==context2
    query = tokenizer.tokenize(q)
    #将答案转化为tag的标注
    tags = [tag_idxs['O']]*len(context2)
    for i,an in enumerate(ans):
        if len(an)==4:#第一轮对话的情况
            start,end,ent_str=an[1:]
        elif len(an)==3:#提取尾实体的信息
            start,end,ent_str=an[-1][1:]
        s1 = [tokenizer.tokenize(t) for t in context[:start].split()]
        #s2 = tokenizer.tokenize(context[:start])
        #assert sum(s1,[])==s2
        new_start = sum([len(s) for s in s1])
        new_end = new_start+len(tokenizer.tokenize(ent_str))-1#这个end是取的闭区间
        if new_start!=new_end:
            tags[new_start]=tag_idxs['B']
            tags[new_end]=tag_idxs['E']
            for i in range(new_start+1,new_end):
                tags[i]=tag_idxs['M']
        else:
            tags[new_start] = tag_idxs['S']
    txt_len = len(query)+len(context2)+3
    if txt_len>max_len:
        context2 = context2[:max_len-len(query)-3]
        tags = tags[:max_len-len(query)-3]
    txt = ['[CLS]']+query+['[SEP]']+context2+['[SEP]']
    txt_ids = tokenizer.convert_tokens_to_ids(txt)
    #[CLS]的tag用来判断是否存在答案
    tags1 = [tag_idxs['O'] if len(ans)>0 else tag_idxs['S']]+[-1]*(len(query)+1)+tags+[-1]
    context_mask = [1]+[0]*(len(query)+1)+[1]*len(context2)+[0]
    token_type_ids = [0]*(len(query)+2)+[1]*(len(context2)+1)
    return txt_ids,tags1,context_mask,token_type_ids


class MyDataset:
    """
    这里假设我们的一个样本是两轮问答，或者一轮问答（第二轮为空）
    这可能导致第一轮问答被多个两轮问答作为首轮问答使用
    """
    def __init__(self,path,tokenizer,max_len=512):
        with open(path,encoding='utf-8') as f:
            data = json.load(f)
        self.all_qas = []
        for d in tqdm(data,desc="dataset"):
            context = d['context']
            qa_pairs = d['qa_pairs']
            t1 = qa_pairs[0]
            t2 = qa_pairs[1]
            qas = []
            t1_qas = []
            t2_qas = []
            dict1 = {}#key:i,value:t1[i][ans],value是一个实体的列表
            dict2 = {}#key:t2[i][ans][0],value:i,
            for i,(q,ans) in enumerate(t1.items()):#这里的ans是某种类型的实体的列表
                txt_ids,tags,context_mask,token_type_ids = get_inputs(context,q,tokenizer,max_len,ans)
                t1_qas.append({"txt_ids":txt_ids,"tags":tags,"context_mask":context_mask,"token_type_ids":token_type_ids})
                dict1[i] = ans
            for i,(q,ans) in enumerate(t2.items()):
                txt_ids,tags,context_mask,token_type_ids = get_inputs(context,q,tokenizer,max_len,ans)
                t2_qas.append({"txt_ids":txt_ids,"tags":tags,"context_mask":context_mask,"token_type_ids":token_type_ids})
                for an in ans:
                    #问题由(head_entity_type,relation_type,end_entity_type)确定，但是这里head/end_entity都没具体确定，所以存在多个答案
                    dict2[tuple(an[1])]=dict2.get(tuple(an[1]),[])+[i]#一个头实体可能有对多个关系的提问
            #建立两轮问答的对关系
            for i,ans1 in dict1.items():
                if len(ans1)>0:
                    for an in ans1:
                        an = tuple(an)
                        if an in dict2:
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

class T1Dataset:
    """暂时只考虑ACE2005"""
    def __init__(self,path,query_templates,tokenizer,window_size,overlap,max_len=512):
        with open(path,encoding="utf=8") as f:
            data = json.load(f)
        self.t1_qas = []
        self.passages = []
        self.entities = []
        self.relations = []
        self.window_size = window_size
        self.overlap= overlap
        self.t1_querys = []
        self.t1_ids = []
        self.t1_gold = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        for ent_type in preprocess.ace2005_entities:
            query = preprocess.get_question(ent_type)
            self.t1_querys.append(query)
        for p_id,d in enumerate(tqdm(data,desc="t1_dataset")):
            passage = d["passage"]
            entities = d['entities']
            relations = d['relations']
            self.passages.append(passage)
            self.entities.append(entities)
            self.relations.append(relations)
            sent,sent_range = preprocess.chunk_passage(passage,window_size,overlap)
            for ent in entities:
                self.t1_gold.append((p_id,ent[1],ent[2]))
            for s_id,s in enumerate(sent):
                for q_id,q in enumerate(self.t1_querys):
                    txt_ids,_,context_mask,token_type_ids = get_inputs(s,q,tokenize,max_len)
                    #context mask对解码有用
                    self.t1_qas.append({"txt_ids":txt_ids,"context_mask":context_mask,"token_type_ids":token_type_ids})
                    self.t1_ids.append((p_id,s_id,q_id))
    
    def __len__(self):
        return len(self.t1_qas)
    
    def __getitem__(self,i):
        return self.t1_qas[i]
    

class T2Dataset:
    def __init__(self,t1_dataset,t1_predict,relation_triple,position_in_query=False):
        """
        Args:
            t1_dataset: 第一轮问答用到的dataset，是上面的T1Dataset类的实例
            t1_predict: 第一轮问答得到的答案,t1_predict[i][j][k]代表第i篇文章，第j个window，第k个问题，对应的答案列表[e1,e2,...],其中ei为(s,e)左闭右开的索引
            relation_triple: 代表我们允许的<entity_type,relation_type,entity_type>三元组
            position_query: 把头实体在context的位置信息编码到query中（因为head entity可能存在实体对应的字符可能出现在多个位置的情况）
        """
        self.t1_dataset = t1_dataset
        tokenizer = t1_dataset.tokenizer
        max_len = t1_dataset.max_len
        self.t2_qas = []
        self.t2_ids = []
        self.t2_gold = []
        for i in range(len(t1_predict)):
            passage = t1_dataset.passages[i]
            for j in range(len(t1_predict[i])):
                context = t1_dataset
                for k in range(len(t1_predict[i][j])):
                    head_entity = None #(head_entity_type,start_idx,end_idx)的三元组
                    relation_type = None
                    for ei in range(len(t1_predict[i][j][k])):
                        end_entity_type = None
                        query = get_question(head_entity,relation_type,end_entity_type,position_in_query=True)
                        txt_ids,_,context_mask,token_type_ids = get_inputs(context,query,tokenizer,max_len)
                        self.t2_qas.append({"txt_ids":txt_ids,"context_mask":context_mask,"token_type_ids":token_type_ids})
                        self.t2_ids.append([i,j,k,ei])
                        
        
    def __len__(self):
        return len(self.t2_qas)
    
    def __getitem__(self,i):
        return self.t2_qas[i]
    

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


def load_data(file_path,batch_size,max_len,pretrained_model_path,dist=False,shuffle=False):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    dataset = MyDataset(file_path,tokenizer,max_len)
    sampler = DistributedSampler(dataset) if dist else None
    dataloader = DataLoader(dataset,batch_size,sampler=sampler,shuffle=shuffle if not sampler else None,collate_fn=collate_fn)
    return dataloader

def load_t1_data():
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    dataset = T1Dataset()

def load_t2_data():
    pass


if __name__=="__main__":
    dataloader = load_data(r'C:\Users\DELL\Desktop\mtqa4kg\data\cleaned_data\ACE2005\2_mini_train.json',2,200,r'C:\Users\DELL\Desktop\chinese-bert-wwm-ext')
    dataloader1 = list(dataloader)