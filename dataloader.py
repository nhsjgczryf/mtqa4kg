from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertTokenizer
from preprocess import passage_blocks, get_question
from constants import *


def collate_fn(batch):
    #这里我们直接就有turn mask
    nbatch = {}
    for b in batch:
        for k,v in b.items():
            nbatch[k] = nbatch.get(k,[]) + [torch.tensor(v)]
    txt_ids = nbatch['txt_ids']
    tags = nbatch['tags']
    context_mask = nbatch['context_mask']
    token_type_ids = nbatch['token_type_ids']
    turn_mask =nbatch['turn_mask']
    # 下面进行padding操作
    ntxt_ids = pad_sequence(txt_ids, batch_first=True, padding_value=0)  # padding的值要在词汇表里面
    ntags = pad_sequence(tags, batch_first=True, padding_value=-1)

    ncontext_mask = pad_sequence(context_mask, batch_first=True, padding_value=0)  # 这个和之前对query的padding一致
    ntoken_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[i,:txt_len] = 1
    nbatch['txt_ids'] = ntxt_ids
    nbatch['tags'] = ntags
    nbatch['context_mask'] = ncontext_mask
    nbatch['token_type_ids'] = ntoken_type_ids
    nbatch['attention_mask'] = attention_mask
    nbatch['turn_mask'] = torch.tensor(turn_mask, dtype=torch.uint8)
    return nbatch


def collate_fn1(batch):
    #这个是在test上预测的时候使用的collate函数
    nbatch = {}
    for b in batch:
        for k, v in b.items():
            nbatch[k] = nbatch.get(k, []) + [torch.tensor(v)]
    txt_ids = nbatch['txt_ids']
    context_mask = nbatch['context_mask']
    token_type_ids = nbatch['token_type_ids']
    ntxt_ids = pad_sequence(txt_ids, batch_first=True, padding_value=0)  # padding的值要在词汇表里面
    ncontext_mask = pad_sequence(context_mask, batch_first=True, padding_value=0)  # 这个和之前对query的padding一致
    ntoken_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[i,:txt_len] = 1
    nbatch['txt_ids'] = ntxt_ids
    nbatch['context_mask'] = ncontext_mask
    nbatch['token_type_ids'] = ntoken_type_ids
    nbatch['attention_mask'] = attention_mask
    return nbatch


def get_inputs(context,q,tokenizer,title="",max_len=200,ans=[]):
    """
    Args:
        context: 已经被tokenize后的上下文
        q:  没有被tokenize的问题
        title(可选参数): 没有被tokeizer的title
        max_len: 允许的最大的长度
        ans: 答案列表
    Returns:
        txt_ids: [CLS]question[SEP]context[SEP]编码后的句子
        tags1: 对应的标注序列
        context_mask： 用来确定context的位置，也即是有tag标注的位置
    """
    query = tokenizer.tokenize(q)
    tags = [tag_idxs['O']]*len(context)
    for i,an in enumerate(ans):
        if len(an)==4:
            start,end,ent_str = an[1:]
        elif len(an)==3:
            start,end,ent_str = an[-1][1:]
        end = end-1#这里我们变成右侧闭区间
        if start!=end:
            tags[start]=tag_idxs['B']
            tags[end]=tag_idxs['E']
            for i in range(start+1,end):
                tags[i]=tag_idxs['M']
        else:
            tags[start]=tag_idxs['S']
    txt_len = len(query)+len(query)+len(context)+4 if title else len(query)+len(context)+3
    if txt_len > max_len:
        context = context[:max_len - len(query) - 3]
        tags = tags[:max_len - len(query) - 3]
    if title:
        txt = ['[CLS]'] + query+['[SEP]'] +title+ ['[SEP]'] + context + ['[SEP]']
    else:
        txt = ['[CLS]'] + query + ['[SEP]'] + context + ['[SEP]']
    txt_ids = tokenizer.convert_tokens_to_ids(txt)
    # [CLS]的tag用来判断是否存在答案
    if not title:
        tags1 = [tag_idxs['O'] if len(ans) > 0 else tag_idxs['S']] + [-1] * (len(query) + 1) + tags + [-1]
        context_mask = [1] + [0] * (len(query) + 1) + [1] * len(context) + [0]
        token_type_ids = [0] * (len(query) + 2) + [1] * (len(context) + 1)
    else:
        tags1 = [tag_idxs['O'] if len(ans) > 0 else tag_idxs['S']] + [-1]*(len(query)+len(title)+2) + tags + [-1]
        context_mask = [1] + [0]*(len(query)+len(title)+2)+ [1] * len(context) + [0]
        token_type_ids = [0]*(len(query)+len(title)+3)+[1]*(len(context) + 1)
    return txt_ids, tags1, context_mask, token_type_ids

def t2_down_sample_prob(t2,down_sample_ratio,epoch,threshold):
    import numpy as np
    np.random.seed(epoch)
    relas = {}
    for q,ans in t2.items():
        #这里暂时只考虑ACE2005
        rel = query2relation(q,ace2005_question_templates)
        relas[q]=rel
    possitive_t2 = {}
    negative_t2 = {}
    for q,ans in t2.items():
        if ans:
            possitive_t2[q]=ans
        else:
            negative_t2[q]=ans
    #注意考虑没有正样本的情况，我们假设存在一个正样本，然后仍然对负样本采样
    n_negative = (1-down_sample_ratio)*max(len(possitive_t2),1)/(down_sample_ratio+1e-10)
    n_negative = min(round(n_negative),len(negative_t2))
    if 0<n_negative<=len(t2):
        #得到每个负样本出现的概率
        neg_prob = {}
        for q,ans in negative_t2.items():
            rel = relas[q]
            idx1,idx2 = rel[0],rel[1:]
            idx1,idx2 = ace2005_idx1[idx1],ace2005_idx2[idx2]
            neg_prob[q]=ace2005_dist[idx1][idx2]
        negative_t2 = list(negative_t2.items())
        neg_prob = [neg_prob[q] for q,ans in negative_t2]
        neg_prob = np.array(neg_prob)
        neg_prob = neg_prob>=threshold
        neg_weight = neg_prob/neg_prob.sum()
        n_negative = min(neg_prob.sum(),n_negative)
        indexs =  np.random.choice(len(negative_t2),n_negative,False,neg_weight)
        negative_t2 = [negative_t2[i] for i in indexs]
        negative_t2 = dict(negative_t2)
    elif n_negative==0:
        negative_t2 = {}
    possitive_t2.update(negative_t2)
    return possitive_t2


def query2relation(question,question_templates):
    '''
    把问题还原为三元组<entity_type,relation_type,entity_type>这个东西
    这个函数的具体实现和我们的模板的规则有关
    '''
    import re
    turn2_questions = question_templates['qa_turn2']
    turn2_questions = {v:k for k,v in turn2_questions.items()}
    for k,v in turn2_questions.items():
        k1 = k.replace("XXX.","")
        if question.startswith(k1):
            return eval(v)

class MyDataset:
    """
    这里我们每个window的负样本有一个下采样的过程， 所以需要迭代更多的epoch，
    并且每个epoch尽量使用不同的随机数种子重新采样
    """

    def __init__(self, path, tokenizer, max_len=512,down_sample_ratio=0.5,epoch=0,threshold=5):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        self.data =data
        self.epoch=epoch
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.down_sample_ratio = down_sample_ratio
        self.threshold = threshold
        self.init_data()

    #下面这个函数最好是每个一个epoch就调用一次，这样可以使我们的采用更有多样性
    def init_data(self):
        self.all_qas = []
        #print(len(self.data))
        for d in tqdm(self.data, desc="dataset"):
            context = d['context']
            title = d['title']
            qa_pairs = d['qa_pairs']
            t1 = qa_pairs[0]
            t2 = qa_pairs[1]
            #t2 = t2_down_sample(t2,self.down_sample_ratio,self.epoch)#debug的时候我们不要采样
            t2 = t2_down_sample_prob(t2,self.down_sample_ratio,self.epoch,self.threshold)
            qas = []
            t1_qas = []
            t2_qas = []
            dict1 = {}  # key:i,value:t1[i][ans],value是一个实体的列表
            dict2 = {}  # key:t2[i][ans][0],value:i,
            for i, (q, ans) in enumerate(t1.items()):  # 这里的ans是某种类型的实体的列表
                txt_ids, tags, context_mask, token_type_ids = get_inputs(context, q, self.tokenizer, title, self.max_len, ans)
                t1_qas.append(
                    {"txt_ids": txt_ids, "tags": tags, "context_mask": context_mask, "token_type_ids": token_type_ids,'turn_mask':0})
                dict1[i] = ans
            for i, (q, ans) in enumerate(t2.items()):
                txt_ids, tags, context_mask, token_type_ids = get_inputs(context, q, self.tokenizer, title, self.max_len, ans)
                t2_qas.append(
                    {"txt_ids": txt_ids, "tags": tags, "context_mask": context_mask, "token_type_ids": token_type_ids, 'turn_mask':1})
                for an in ans:
                    # 问题由(head_entity_type,relation_type,end_entity_type)确定，但是这里head/end_entity都没具体确定，所以存在多个答案
                    dict2[tuple(an[1])] = dict2.get(tuple(an[1]), []) + [i]  # 一个头实体可能有对多个关系的提问
            for qa in t1_qas+t2_qas:
                qas.append(qa)
            self.all_qas.extend(qas)

    def set_epoch(self,epoch):
        self.epoch = epoch

    def __len__(self):
        # 返回有多少个完整的问答（可能是一轮，也可能是两轮）
        return len(self.all_qas)

    def __getitem__(self, i):
        return self.all_qas[i]


class T1Dataset:
    '''暂时只考虑ACE2005'''

    def __init__(self, test_path, tokenizer, window_size, overlap, max_len=512):
        with open(test_path, encoding="utf=8") as f:
            data = json.load(f)
        self.t1_qas = []
        self.passages = []
        self.entities = []
        self.relations = []
        self.titles = []
        self.window_size = window_size
        self.overlap = overlap
        self.t1_querys = []
        self.t1_ids = []
        self.t1_gold = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.passage_windows = []  # passage_windows[i][j]代表第i篇文章的第j个window
        self.query_offset1 = []
        self.window_offset_base = window_size-overlap
        for ent_type in ace2005_entities:
            query = get_question(ace2005_question_templates,ent_type)
            self.t1_querys.append(query)
        for p_id, d in enumerate(tqdm(data, desc="t1_dataset")):
            passage = d["passage"]
            entities = d['entities']
            relations = d['relations']
            title = d['title']
            self.passages.append(passage)
            self.entities.append(entities)
            self.relations.append(relations)
            self.titles.append(title)
            blocks, regions = passage_blocks(passage, window_size, overlap)
            self.passage_windows.append(blocks)
            for ent in entities:
                self.t1_gold.append((p_id, tuple(ent[:-1])))
            for b_id, block in enumerate(blocks):
                for q_id, q in enumerate(self.t1_querys):
                    txt_ids, _, context_mask, token_type_ids = get_inputs(block, q, tokenizer,title, max_len)
                    # context mask对解码有用
                    self.t1_qas.append({"txt_ids": txt_ids, "context_mask": context_mask,
                                        "token_type_ids": token_type_ids})
                    self.t1_ids.append((p_id, b_id, ace2005_entities[q_id]))
                    ofs = len(title)+len(tokenizer.tokenize(q))+3
                    self.query_offset1.append(ofs)

    def __len__(self):
        return len(self.t1_qas)

    def __getitem__(self, i):
        return self.t1_qas[i]


class old_T2Dataset:

    def __init__(self, t1_dataset, t1_predict,threshold=5):
        '''
        Args:
            t1_dataset: 第一轮问答用到的dataset，是上面的T1Dataset类的实例
            t1_predict: 第一轮问答得到的答案，t1_predict[i]代表dataset1中的(p_idi,s_idi,q_idi)对应的样本的答案。(未修正，是window种的索引)
        '''
        tokenizer = t1_dataset.tokenizer
        max_len = t1_dataset.max_len
        t1_ids = t1_dataset.t1_ids
        passages = t1_dataset.passages
        titles = t1_dataset.titles
        passage_windows = t1_dataset.passage_windows
        self.t2_qas = []
        self.t2_ids = []
        self.t2_gold = []
        self.query_offset2 = []
        relations = t1_dataset.relations
        entities = t1_dataset.entities
        query_offset1 = t1_dataset.query_offset1
        window_offset_base = t1_dataset.window_offset_base
        for passage_id, (ents, rels) in enumerate(zip(entities, relations)):
            for re in rels:
                head, rel, end = ents[re[1]], re[0], ents[re[2]]
                self.t2_gold.append((passage_id, (tuple(head[:-1]), rel, tuple(end[:-1]))))
        for i,(_id,pre) in enumerate(zip(tqdm(t1_ids,desc="t2 dataset"),t1_predict)):
            passage_id, window_id, head_entity_type = _id
            window_offset = window_offset_base*window_id
            context = passage_windows[passage_id][window_id]
            title = titles[passage_id]
            head_entities = []
            for start,end in pre:
                start1,end1 = start-query_offset1[i]+window_offset,end-query_offset1[i]+window_offset
                ent_str = tokenizer.convert_tokens_to_string(passages[passage_id][start1:end1])
                head_entity = (head_entity_type, start1, end1,ent_str)
                head_entities.append(head_entity)
            for head_entity in head_entities:
                for rel in ace2005_relations:
                    for end_ent_type in ace2005_entities:
                        # 这里暂时考虑所有的情况
                        idx1,idx2 = ace2005_idx1[head_entity[0]],ace2005_idx2[(rel,end_ent_type)]
                        if ace2005_dist[idx1][idx2]>=threshold:
                            query = get_question(ace2005_question_templates,head_entity, rel, end_ent_type)
                            txt_ids, _, context_mask, token_type_ids = get_inputs(context, query, tokenizer, title, max_len)
                            self.t2_qas.append({"txt_ids": txt_ids, "context_mask": context_mask,
                                                "token_type_ids": token_type_ids})
                            self.t2_ids.append((passage_id, window_id, head_entity[:-1], rel, end_ent_type))
                            ofs = len(title) + len(tokenizer.tokenize(query)) + 3
                            self.query_offset2.append(ofs)

    def __len__(self):
        return len(self.t2_qas)

    def __getitem__(self, i):
        return self.t2_qas[i]



class T2Dataset:
    def __init__(self, t1_dataset, t1_predict,threshold=5,max_distance=100):
        '''
        Args:
            t1_dataset: 第一轮问答用到的dataset，是上面的T1Dataset类的实例
            t1_predict: 第一轮问答得到的答案，t1_predict[i]代表dataset1中的(p_idi,s_idi,q_idi)对应的样本的答案。(未修正，是window种的索引)
        '''
        tokenizer = t1_dataset.tokenizer
        max_len = t1_dataset.max_len
        t1_ids = t1_dataset.t1_ids
        passages = t1_dataset.passages
        titles = t1_dataset.titles
        passage_windows = t1_dataset.passage_windows
        self.t2_qas = []
        self.t2_ids = []
        self.t2_gold = []
        self.query_offset2 = []
        relations = t1_dataset.relations
        entities = t1_dataset.entities
        query_offset1 = t1_dataset.query_offset1
        window_offset_base = t1_dataset.window_offset_base
        for passage_id, (ents, rels) in enumerate(zip(entities, relations)):
            for re in rels:
                head, rel, end = ents[re[1]], re[0], ents[re[2]]
                self.t2_gold.append((passage_id, (tuple(head[:-1]), rel, tuple(end[:-1]))))
        if max_distance>0:
            #对所有的predict，添加对应的str信息
            t1_predict1 = []
            t1_ids1 = []
            for i,(_id, pre) in enumerate(zip(t1_ids,t1_predict)):
                passage_id,window_id= _id[0],_id[1]
                window_offset = window_id*window_offset_base
                pre1 = []
                for start, end in pre:
                    start1,end1 = start-query_offset1[i]+window_offset, end-query_offset1[i]+window_offset
                    ent_str = tokenizer.convert_tokens_to_string(passages[passage_id][start1:end1])
                    t1_predict1.append((start1,end1,ent_str))
                    t1_ids1.append(_id)
            #这里我们考虑按照(passage_id,window_id)进行聚类，因为我们需要根据window_id进行聚类
            ids_and_predict = list(zip(t1_ids1,t1_predict1))
            dict1 = {(_id[0],_id[1]):[] for _id,pre in ids_and_predict}
            for _id,pre in ids_and_predict:
                dict1[(_id[0],_id[1])].append((_id,pre))
            for (passage_id,window_id),_ids_and_pres in dict1.items():
                    #对同一个window内的预测的实体按照start的索引进行排序
                    _ids_and_pres = sorted(_ids_and_pres,key=lambda x: x[1][0])
                    context =  passage_windows[passage_id][window_id]
                    title = titles[passage_id]
                    for i, (_id1,pre1) in enumerate(_ids_and_pres):
                        start = pre1[0]
                        for j,(_id2,pre2) in enumerate(_ids_and_pres[i+1:],i+1):
                            if pre2[0]>start+max_distance:
                                break
                            else:
                                head_type, end_type = _id1[-1],_id2[-1]
                                for rel_type in ace2005_relations:
                                    idx1,idx2 =ace2005_idx1[head_type],ace2005_idx2[(rel_type,end_type)]
                                    if ace2005_dist[idx1][idx2]>=threshold:
                                        head_entity = (_id1[-1],pre1[0],pre1[1],pre1[2])
                                        query = get_question(ace2005_question_templates,head_entity,rel_type,end_type)
                                        txt_ids, _, context_mask, token_type_ids = get_inputs(context, query, tokenizer, title, max_len)
                                        self.t2_qas.append({"txt_ids": txt_ids, "context_mask": context_mask,
                                                            "token_type_ids": token_type_ids})
                                        self.t2_ids.append((passage_id, window_id, head_entity[:-1], rel_type, end_type))
                                        ofs = len(title) + len(tokenizer.tokenize(query)) + 3
                                        self.query_offset2.append(ofs)
        else:
            #这个是最初的不考虑第一阶段中window预测得到的实体
            for i,(_id,pre) in enumerate(zip(tqdm(t1_ids,desc="t2 dataset"),t1_predict)):
                passage_id, window_id, head_entity_type = _id
                window_offset = window_offset_base*window_id
                context = passage_windows[passage_id][window_id]
                title = titles[passage_id]
                head_entities = []
                for start,end in pre:
                    start1,end1 = start-query_offset1[i]+window_offset,end-query_offset1[i]+window_offset
                    ent_str = tokenizer.convert_tokens_to_string(passages[passage_id][start1:end1])
                    head_entity = (head_entity_type, start1, end1,ent_str)
                    head_entities.append(head_entity)
                for head_entity in head_entities:
                    for rel in ace2005_relations:
                        for end_ent_type in ace2005_entities:
                            # 这里暂时考虑所有的情况
                            idx1,idx2 = ace2005_idx1[head_entity[0]],ace2005_idx2[(rel,end_ent_type)]
                            if ace2005_dist[idx1][idx2]>=threshold:
                                query = get_question(ace2005_question_templates,head_entity, rel, end_ent_type)
                                txt_ids, _, context_mask, token_type_ids = get_inputs(context, query, tokenizer, title, max_len)
                                self.t2_qas.append({"txt_ids": txt_ids, "context_mask": context_mask,
                                                    "token_type_ids": token_type_ids})
                                self.t2_ids.append((passage_id, window_id, head_entity[:-1], rel, end_ent_type))
                                ofs = len(title) + len(tokenizer.tokenize(query)) + 3
                                self.query_offset2.append(ofs)

    def __len__(self):
        return len(self.t2_qas)

    def __getitem__(self, i):
        return self.t2_qas[i]


class T2Dataset_Small_Window:

    def __init__(self, t1_dataset, t1_predict,threshold=5,left_offset,right_offset):
        '''
        Args:
            t1_dataset: 第一轮问答用到的dataset，是上面的T1Dataset类的实例
            t1_predict: 第一轮问答得到的答案，t1_predict[i]代表dataset1中的(p_idi,s_idi,q_idi)对应的样本的答案。(未修正，是window种的索引)
        '''
        tokenizer = t1_dataset.tokenizer
        max_len = t1_dataset.max_len
        t1_ids = t1_dataset.t1_ids
        passages = t1_dataset.passages
        titles = t1_dataset.titles
        passage_windows = t1_dataset.passage_windows
        self.t2_qas = []
        self.t2_ids = []
        self.t2_gold = []
        self.query_offset2 = []
        relations = t1_dataset.relations
        entities = t1_dataset.entities
        query_offset1 = t1_dataset.query_offset1
        window_offset_base = t1_dataset.window_offset_base
        for passage_id, (ents, rels) in enumerate(zip(entities, relations)):
            for re in rels:
                head, rel, end = ents[re[1]], re[0], ents[re[2]]
                self.t2_gold.append((passage_id, (tuple(head[:-1]), rel, tuple(end[:-1]))))
        for i,(_id,pre) in enumerate(zip(tqdm(t1_ids,desc="t2 dataset"),t1_predict)):
            passage_id, window_id, head_entity_type = _id
            window_offset = window_offset_base*window_id
            context = passage_windows[passage_id][window_id]
            title = titles[passage_id]
            head_entities = []
            for start,end in pre:
                start1,end1 = start-query_offset1[i]+window_offset,end-query_offset1[i]+window_offset
                ent_str = tokenizer.convert_tokens_to_string(passages[passage_id][start1:end1])
                head_entity = (head_entity_type, start1, end1,ent_str)
                head_entities.append(head_entity)
            for head_entity in head_entities:
                for rel in ace2005_relations:
                    for end_ent_type in ace2005_entities:
                        # 这里暂时考虑所有的情况
                        idx1,idx2 = ace2005_idx1[head_entity[0]],ace2005_idx2[(rel,end_ent_type)]
                        if ace2005_dist[idx1][idx2]>=threshold:
                            query = get_question(ace2005_question_templates,head_entity, rel, end_ent_type)
                            txt_ids, _, context_mask, token_type_ids = get_inputs(context, query, tokenizer, title, max_len)
                            self.t2_qas.append({"txt_ids": txt_ids, "context_mask": context_mask,
                                                "token_type_ids": token_type_ids})
                            self.t2_ids.append((passage_id, window_id, head_entity[:-1], rel, end_ent_type))
                            ofs = len(title) + len(tokenizer.tokenize(query)) + 3
                            self.query_offset2.append(ofs)

    def __len__(self):
        return len(self.t2_qas)

    def __getitem__(self, i):
        return self.t2_qas[i]



def load_data(file_path, batch_size, max_len, pretrained_model_path, dist=False, shuffle=False,down_sample_ratio=0.5,threshold=5):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    dataset = MyDataset(file_path, tokenizer, max_len,down_sample_ratio,threshold)
    sampler = DistributedSampler(dataset) if dist else None
    dataloader = DataLoader(dataset, batch_size, sampler=sampler, shuffle=shuffle if not sampler else False,
                            collate_fn=collate_fn)
    return dataloader

def reload_data(old_dataloader,batch_size,max_len,down_sample_ratio,threshold,local_rank,shuffle=True):
    """这里我们根据原来保存的dataloader，重新导入"""
    dataset = old_dataloader.dataset
    old_max_len,old_down_sample_ratio,old_threshold = dataset.max_len, dataset.down_sample_ratio,dataset.threshold
    if not( old_max_len==max_len and old_down_sample_ratio==down_sample_ratio and old_threshold==threshold):
        dataset.max_len =max_len
        dataset.down_sample_ratio = down_sample_ratio
        dataset.threshold = threshold
        dataset.init_data()
    sampler = DistributedSampler(dataset,rank=local_rank) if local_rank!=-1 else None
    dataloader = DataLoader(dataset,batch_size,sampler=sampler, shuffle=shuffle,collate_fn=collate_fn)
    return dataloader


def load_t1_data(test_path, pretrained_model_path, window_size, overlap, batch_size=10, max_len=512):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    t1_dataset = T1Dataset(test_path, tokenizer, window_size, overlap,max_len)
    dataloader = DataLoader(t1_dataset, batch_size, collate_fn=collate_fn1)
    return dataloader


def load_t2_data(t1_dataset, t1_predict, batch_size=10, threshold=5, max_distance=100):
    t2_dataset = T2Dataset(t1_dataset, t1_predict, threshold, max_distance)
    dataloader = DataLoader(t2_dataset, batch_size, collate_fn=collate_fn1)
    return dataloader


if __name__ == "__main__":
    dataloader = load_data(r'C:\Users\DELL\Desktop\mtqa4kg\data\cleaned_data\ACE2005\\\bert_base_uncased\1_mini_train.json', 2, 200,
                           r'bert-base-uncased')
    dataloader1 = list(dataloader)