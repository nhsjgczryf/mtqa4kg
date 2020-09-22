import json
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import  pad_sequence
from torch.utils.data import DataLoader,DistributedSampler
import random
from transformers import BertTokenizer
from preprocess import chunk_passage,get_question

tag_idxs = {'B':0,'M':1,'E':2,'S':3,'O':4}
ace2005_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2005_relations = ['ART', 'GEN-AFF', 'ORG-AFF', 'PART-WHOLE', 'PER-SOC', 'PHYS']


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

def collate_fn1(batch):
    nbatch = {}
    for b in batch:
        for k,v in b.items():
            nbatch[k]=nbatch.get(k,[])+[torch.tensor(v)]
    txt_ids = nbatch['txt_ids']
    tags = nbatch['tags']
    context_mask = nbatch['context_mask']
    token_type_ids = nbatch['token_type_ids']
    ntxt_ids = pad_sequence(txt_ids,batch_first=True,padding_value=0)#padding的值要在词汇表里面
    ncontext_mask = pad_sequence(context_mask,batch_first=True,padding_value=0)#这个和之前对query的padding一致
    ntoken_type_ids = pad_sequence(token_type_ids,batch_first=True,padding_value=1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[:txt_len]=1
    ntags = pad_sequence(tags,batch_first=True,padding_value=-1)
    nbatch['tags']=ntags
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


def get_gold_ans(passgae,tokenizer,ans):
    """
    Args:
        passage:没有被切分的原文
    """

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
                #print('ORG: [CLS]'+q+'[SEP]'+context+'[SEP]')
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
                            #如果turn1的答案作为turn2的头实体
                            for j in dict2[an]:
                                qas.append([t1_qas[i],t2_qas[j]])
                        else:
                            #如果turn1的答案并不参与关系
                            qas.append([t1_qas[i]])
                else:
                    #如果turn1没有答案
                    qas.append([t1_qas[i]])
            self.all_qas.extend(qas)

    def __len__(self):
        #返回有多少个完整的问答（可能是一轮，也可能是两轮）
        return len(self.all_qas)

    def __getitem__(self, i):
        return self.all_qas[i]

def wordpice_idx(window,wordpice_str):
    """
    将wordpiece tokenizer还原的string和原来的string对齐
    wordpiece_str会比原来的string多一些空白符号
    Returns:
        wordpiece_str在window中的末位置的下一个位置
    """
    #处理uncased的tokenizer情况
    window=window.lower()
    wordpice_str=wordpice_str.lower()
    idx1=0
    idx2=0
    while idx2<len(wordpice_str):
        if window[idx1]==wordpice_str[idx2]:
            idx1+=1
            idx2+=1
        elif wordpice_str[idx2]==" ":
            idx2+=1
        elif window[idx1]==" ":
            idx1+=1
        else:
            raise Exception("无法匹配:\n{}\n{}".format(window,wordpice_str))
    assert window[:idx1].replace(" ","")==wordpice_str.replace(" ","")
    return idx1

class T1Dataset:
    '''暂时只考虑ACE2005'''
    """
    def __init__(self,test_path,tokenizer,window_size,overlap,max_len=512):
        with open(test_path,encoding="utf=8") as f:
            data = json.load(f)
        self.t1_qas = []
        self.passages = []
        self.entities = []
        self.relations = []
        self.window_size = window_size
        self.overlap= overlap
        self.t1_querys = []
        self.t1_ids = []
        self.t1_gold = []#注意这里的gold中的(s,e)是在[CLS]query[SEP]window[SEP]中的便宜
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.passage_windows= [] #passage_windows[i][j]代表第i篇文章的第j个window
        self.query_offset1 = {}
        for ent_type in ace2005_entities:
            query = get_question(ent_type)
            self.t1_querys.append(query)
            offset = len(tokenizer.tokenize(query))+2
            self.query_offset1[ent_type]=offset
        for p_id,d in enumerate(tqdm(data,desc="t1_dataset")):
            passage = d["passage"]
            entities = d['entities']
            relations = d['relations']
            self.passages.append(passage)
            self.entities.append(entities)
            self.relations.append(relations)
            sent, _ = chunk_passage(passage,window_size,overlap)
            self.passage_windows.append(sent)
            for ent in entities:
                self.t1_gold.append((p_id,tuple(ent)[:-1]))
            for s_id,s in enumerate(sent):
                for q_id,q in enumerate(self.t1_querys):
                    #print('T1: [CLS]'+q+'[SEP]'+s+'[SEP]')
                    txt_ids,tags,context_mask,token_type_ids = get_inputs(s,q,tokenizer,max_len)
                    #context mask对解码有用
                    self.t1_qas.append({"txt_ids":txt_ids,"tags":tags,"context_mask":context_mask,"token_type_ids":token_type_ids})
                    self.t1_ids.append((p_id,s_id,ace2005_entities[q_id]))
    """
    def __init__(self,test_path,tokenizer,window_size,overlap,max_len=512):
        with open(test_path,encoding="utf=8") as f:
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
        self.passage_windows= [] #passage_windows[i][j]代表第i篇文章的第j个window
        self.query_offset1 = {}
        self.window_offset1 = []
        for ent_type in ace2005_entities:
            query = get_question(ent_type)
            self.t1_querys.append(query)
            offset = len(tokenizer.tokenize(query))+2
            self.query_offset1[ent_type]=offset
        for p_id,d in enumerate(tqdm(data,desc="t1_dataset")):
            passage = d["passage"]
            entities = d['entities']
            relations = d['relations']
            self.passages.append(passage)
            self.entities.append(entities)
            self.relations.append(relations)
            sent, regions = chunk_passage(passage,window_size,overlap)
            self.passage_windows.append(sent)
            for ent in entities:
                self.t1_gold.append((p_id,tuple(ent)))
            for s_id,s in enumerate(sent):
                region = regions[s_id]
                soffset = region[0]#当前window的offset
                for q_id,q in enumerate(self.t1_querys):
                    #print('T1: [CLS]'+q+'[SEP]'+s+'[SEP]')
                    #这里的tags是历史遗留，没有任何用处
                    txt_ids,tags,context_mask,token_type_ids = get_inputs(s,q,tokenizer,max_len)
                    #context mask对解码有用
                    self.t1_qas.append({"txt_ids":txt_ids,"tags":tags,"context_mask":context_mask,"token_type_ids":token_type_ids})
                    self.t1_ids.append((p_id,s_id,ace2005_entities[q_id]))
                    self.window_offset1.append(soffset)
                    
    def __len__(self):
        return len(self.t1_qas)
    
    def __getitem__(self,i):
        return self.t1_qas[i]
    

class T2Dataset:
    """
    def __init__(self,t1_dataset,t1_predict,position_in_query=False):
        '''
        Args:
            t1_dataset: 第一轮问答用到的dataset，是上面的T1Dataset类的实例
            t1_predict: 第一轮问答得到的答案，t1_predict[i]代表dataset1中的(p_idi,s_idi,q_idi)对应的样本的答案。(未修正，是window种的索引)
            position_query: 把头实体在context的位置信息编码到query中（因为head entity可能存在实体对应的字符可能出现在多个位置的情况）【这里先假设一个实体字符串+实体类型能够确定一个实体的位置，所以这个参数暂时不考虑】
        '''
        tokenizer = t1_dataset.tokenizer
        max_len = t1_dataset.max_len
        t1_ids = t1_dataset.t1_ids
        passages = t1_dataset.passages
        passage_windows = t1_dataset.passage_windows
        window_size = t1_dataset.window_size
        overlap = t1_dataset.overlap
        self.t2_qas = []
        self.t2_ids = []
        self.t2_gold = []
        self.query_offset2 = []
        relations = t1_dataset.relations
        entities = t1_dataset.entities
        query_offset1=t1_dataset.query_offset1
        for passage_id, (ents,rels) in enumerate(zip(entities,relations)):
            for re in rels:
                head,rel,end = ents[re[1]],re[0],ents[re[2]]
                self.t2_gold.append((passage_id,(tuple(head[:-1]),rel,tuple(end[:-1]))))
        for t1_id, t1_pre in tqdm(zip(t1_ids,t1_predict),desc="t2_dataset"):
            passage_id,window_id,head_entity_type = t1_id
            offset1 = (window_size-overlap)*window_id
            offset2 = query_offset1[head_entity_type]
            window = passage_windows[passage_id][window_id]
            head_entities = []
            for s,e in t1_pre:
                ns,ne = s-offset1-offset2,e-offset1-offset2
                head_entity_str = passages[passage_id][ns:ne]
                head_entity = (head_entity_type,ns,ne,head_entity_str)
                head_entities.append(head_entity)
            for head_entity in head_entities:
                for rel in ace2005_relations:
                    for end_ent_type in ace2005_entities:
                        #这里暂时考虑所有可能的情况
                        #if (head_entity_type,rel,end_ent_type) in preprocess.ace2005_relation_triples:
                        query = get_question(head_entity,rel,end_ent_type)
                        txt_ids,tags,context_mask,token_type_ids = get_inputs(window,query,tokenizer,max_len)
                        self.t2_qas.append({"txt_ids":txt_ids,"tags":tags,"context_mask":context_mask,"token_type_ids":token_type_ids})
                        self.t2_ids.append((passage_id,window_id,head_entity[:-1],rel,end_ent_type))
                        self.query_offset2.append(len(tokenizer.tokenize(query))+2)
    """
    def __init__(self,t1_dataset,t1_predict,position_in_query=False):
        '''
        Args:
            t1_dataset: 第一轮问答用到的dataset，是上面的T1Dataset类的实例
            t1_predict: 第一轮问答得到的答案，t1_predict[i]代表dataset1中的(p_idi,s_idi,q_idi)对应的样本的答案。(未修正，是window种的索引)
            position_query: 把头实体在context的位置信息编码到query中（因为head entity可能存在实体对应的字符可能出现在多个位置的情况）【这里先假设一个实体字符串+实体类型能够确定一个实体的位置，所以这个参数暂时不考虑】
        '''
        tokenizer = t1_dataset.tokenizer
        max_len = t1_dataset.max_len
        t1_ids = t1_dataset.t1_ids
        passages = t1_dataset.passages
        passage_windows = t1_dataset.passage_windows
        window_size = t1_dataset.window_size
        overlap = t1_dataset.overlap
        self.t2_qas = []
        self.t2_ids = []
        self.t2_gold = []
        self.query_offset2 = []
        self.window_offset2 = []
        relations = t1_dataset.relations
        entities = t1_dataset.entities
        query_offset1 = t1_dataset.query_offset1
        window_offset1 = t1_dataset.window_offset1
        for passage_id, (ents,rels) in enumerate(zip(entities,relations)):
            for re in rels:
                head,rel,end = ents[re[1]],re[0],ents[re[2]]
                self.t2_gold.append((passage_id,(tuple(head),rel,tuple(end))))
        for i in tqdm(range(len(t1_ids)),desc="t2_dataset"):
            t1_id = t1_ids[i]
            t1_pre = t1_predict[i]
            passage_id,window_id,head_entity_type = t1_id
            context = passage_windows[passage_id][window_id]
            window = tokenizer.tokenize(context)
            woffset = window_offset1[i]#窗口的偏移
            head_entities = []
            for start,end in t1_pre:
                start1,end1 = start-query_offset1[head_entity_type],end-query_offset1[head_entity_type]
                wordpiece_str = tokenizer.convert_tokens_to_string(window[:start1])
                start2 = wordpice_idx(context,wordpiece_str)+1 if wordpiece_str else 0
                ent_str = tokenizer.convert_tokens_to_string(window[start1:end1])
                end2 = start2 + wordpice_idx(context[start2:],ent_str)
                #end2 = start2+len(ent_str)
                start3,end3 = start2+woffset,end2+woffset
                assert ent_str.lower().replace(" ","")==passages[passage_id][start3:end3].lower().replace(" ","")
                #注意我们评估的时候是大小写敏感的，所以只能使用原文里字符串
                head_entity = (head_entity_type,start3,end3,passages[passage_id][start3:end3])
                head_entities.append(head_entity)
            for head_entity in head_entities:
                for rel in ace2005_relations:
                    for end_ent_type in ace2005_entities:
                        #这里暂时考虑所有的情况
                        #if (head_entity_type,rel,end_ent_type) in preprocess.ace2005_relation_triples:
                        query = get_question(head_entity,rel,end_ent_type)
                        #这里的tags一样是历史遗留，没用
                        txt_ids,tags,context_mask,token_type_ids = get_inputs(context,query,tokenizer,max_len)
                        self.t2_qas.append({"txt_ids":txt_ids,"tags":tags,"context_mask":context_mask,"token_type_ids":token_type_ids})
                        self.t2_ids.append((passage_id,window_id,head_entity,rel,end_ent_type))
                        self.query_offset2.append(len(tokenizer.tokenize(query))+2)
                        self.window_offset2.append(woffset)
        '''
        for t1_id, t1_pre in tqdm(zip(t1_ids,t1_predict),desc="t2_dataset",total=len(t1_ids)):
            passage_id,window_id,head_entity_type = t1_id
            offset1 = (window_size-overlap)*window_id
            offset2 = query_offset1[head_entity_type]
            window = passage_windows[passage_id][window_id]
            head_entities = []
            for s,e in t1_pre:
                ns,ne = s-offset1-offset2,e-offset1-offset2
                head_entity_str = passages[passage_id][ns:ne]
                head_entity = (head_entity_type,ns,ne,head_entity_str)
                head_entities.append(head_entity)
            for head_entity in head_entities:
                for rel in ace2005_relations:
                    for end_ent_type in ace2005_entities:
                        #这里暂时考虑所有可能的情况
                        #if (head_entity_type,rel,end_ent_type) in preprocess.ace2005_relation_triples:
                        query = get_question(head_entity,rel,end_ent_type)
                        txt_ids,tags,context_mask,token_type_ids = get_inputs(window,query,tokenizer,max_len)
                        self.t2_qas.append({"txt_ids":txt_ids,"tags":tags,"context_mask":context_mask,"token_type_ids":token_type_ids})
                        self.t2_ids.append((passage_id,window_id,head_entity[:-1],rel,end_ent_type))
                        self.query_offset2.append(len(tokenizer.tokenize(query))+2)
                        self.window_offset2.append()
        '''
        
    def __len__(self):
        return len(self.t2_qas)
    
    def __getitem__(self,i):
        return self.t2_qas[i]


def load_data(file_path,batch_size,max_len,pretrained_model_path,dist=False,shuffle=False):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    dataset = MyDataset(file_path,tokenizer,max_len)
    sampler = DistributedSampler(dataset) if dist else None
    dataloader = DataLoader(dataset,batch_size,sampler=sampler,shuffle=shuffle if not sampler else False,collate_fn=collate_fn)
    return dataloader

def load_t1_data(test_path,pretrained_model_path,window_size,overlap,batch_size=10,max_len=512):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    t1_dataset = T1Dataset(test_path,tokenizer,window_size,overlap)
    dataloader = DataLoader(t1_dataset,batch_size,collate_fn=collate_fn1)
    return dataloader

def load_t2_data(t1_dataset,t1_predict,batch_size=10):
    t2_dataset = T2Dataset(t1_dataset,t1_predict)
    dataloader = DataLoader(t2_dataset,batch_size,collate_fn=collate_fn1)
    return dataloader


if __name__=="__main__":
    dataloader = load_data(r'C:\Users\DELL\Desktop\mtqa4kg\data\cleaned_data\ACE2005\2_mini_train.json',2,200,r'C:\Users\DELL\Desktop\chinese-bert-wwm-ext')
    dataloader1 = list(dataloader)