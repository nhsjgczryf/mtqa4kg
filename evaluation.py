from tqdm import tqdm
from dataloader import tag_idxs,load_t2_data,wordpice_idx
import torch



def get_score(gold_set,predict_set):
    """得到两个集合的precision,recall.f1"""
    TP = len(set.intersection(gold_set,predict_set))
    precision = TP/(len(predict_set)+1e-6)
    recall = TP/(len(gold_set)+1e-6)
    f1 = 2*precision*recall/(precision+recall+1e-6)
    return precision,recall,f1


def dev_evaluation(model,dataloader):
    """验证集上的评估直接当作NER任务的评估来做，不考虑多轮问答"""
    if hasattr(model,'module'):
        model = model.module
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    gold = []
    predict = []
    tqdm_dataloader = tqdm(dataloader,desc="dev eval")
    with torch.no_grad():
        for i,batch in enumerate(tqdm_dataloader):
            txt_ids, attention_mask, token_type_ids, context_mask, turn_mask,tags=batch['txt_ids'],batch['attention_mask'],batch['token_type_ids'],\
                                                                           batch['context_mask'],batch['turn_mask'],batch['tags']
            tag_idxs = model(txt_ids.to(device), attention_mask.to(device), token_type_ids.to(device))
            predict_spans = tag_decode(tag_idxs,context_mask)
            gold_spans = tag_decode(tags)
            turn1_predict = [p for i,p in enumerate(predict_spans) if turn_mask[i]==0]
            turn2_predict = [p for i,p in enumerate(predict_spans) if turn_mask[i]==1]
            #print("turn1 predict",len(turn1_predict),turn1_predict)
            #print("turn2 predict",len(turn2_predict),turn2_predict)
            #print("dev gold spans:",gold_spans)
            predict.append((i,predict_spans))
            gold.append((i,gold_spans))
    gold2 = set()
    predict2 = set()
    for g in gold:
        i,gold_spans = g
        for j,gs in enumerate(gold_spans):
            for gsi in gs:
                item = (i,j,gsi[0],gsi[1])
                gold2.add(item)
    for p in predict:
        i,pre_spans = p
        for j, ps in enumerate(pre_spans):
            for psi in ps:
                item = (i,j, psi[0],psi[1])
                predict2.add(item)
    precision,recall,f1 = get_score(gold2,predict2)
    return precision,recall,f1

def test_evaluation(model,t1_dataloader):
    if hasattr(model,'module'):
        model = model.module
    model.eval()
    t1_predict = []
    #1_gold = []
    t2_predict = []
    #2_gold = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #第一轮问答
    with torch.no_grad():
        for i,batch in enumerate(tqdm(t1_dataloader,desc="t1 predict")):
            #txt_ids,attention_mask,token_type_ids,context_mask,tags = batch['txt_ids'],batch['attention_mask'],batch['token_type_ids'],batch['context_mask'],batch['tags']
            txt_ids,attention_mask,token_type_ids,context_mask = batch['txt_ids'],batch['attention_mask'],batch['token_type_ids'],batch['context_mask']
            tag_idxs = model(txt_ids.to(device),attention_mask.to(device),token_type_ids.to(device))
            predict_spans = tag_decode(tag_idxs,context_mask)
            #gold_spans = tag_decode(tags)
            t1_predict.extend(predict_spans)
            #print("t1 predict spans:",predict_spans)
            #1_gold.extend(gold_spans)
    #进行第二轮问答
    t2_dataloader = load_t2_data(t1_dataloader.dataset,t1_predict)
    with torch.no_grad():
        for i,batch in enumerate(tqdm(t2_dataloader,desc="t2 predict")):
            #txt_ids,attention_mask,token_type_ids,context_mask,tags = batch['txt_ids'],batch['attention_mask'],batch['token_type_ids'],batch['context_mask'],batch['tags']
            txt_ids,attention_mask,token_type_ids,context_mask = batch['txt_ids'],batch['attention_mask'],batch['token_type_ids'],batch['context_mask']
            tag_idxs = model(txt_ids.to(device),attention_mask.to(device),token_type_ids.to(device))
            predict_spans = tag_decode(tag_idxs,context_mask)
            #gold_spans = tag_decode(tags)
            t2_predict.extend(predict_spans)
            #t2_gold.extend(gold_spans)
    #获取一些需要需要的信息
    t1_ids = t1_dataloader.dataset.t1_ids
    t2_ids = t2_dataloader.dataset.t2_ids
    window_offset1 = t1_dataloader.dataset.window_offset1
    window_offset2 = t2_dataloader.dataset.window_offset2
    tokenizer = t1_dataloader.dataset.tokenizer
    passages = t1_dataloader.dataset.passages
    passage_windows = t1_dataloader.dataset.passage_windows
    query_offset1 = t1_dataloader.dataset.query_offset1
    query_offset2 = t2_dataloader.dataset.query_offset2
    t1_gold = t1_dataloader.dataset.t1_gold
    t2_gold = t2_dataloader.dataset.t2_gold
    #第一阶段的评估，即评估我们的ner的结果
    p1,r1,f1 = eval_t1(tokenizer,passages,passage_windows,t1_gold,t1_predict,t1_ids,window_offset1,query_offset1)
    #第二阶段的评估，即评估我们的ner+re的综合结果
    p2,r2,f2 = eval_t2(tokenizer,passages,passage_windows,t2_gold,t2_predict,t2_ids,window_offset2,query_offset2)
    #p2,r2,f2 = eval_t2(t2_ids, query_offset2,t2_predict,t2_gold,window_size,overlap)
    return (p1,r1,f1),(p2,r2,f2)


def eval_t1(tokenizer,passages,passage_windows,t1_gold,t1_predict,t1_ids,window_offset,query_offset):
    """
    把t1_predict转化为passage上的char level的标注，然后评估。（不用wordpiece level的标注是因为滑窗切分处的tokenzie有可能不一致）
    Args:
        passages(list): 文章。主要用来校验我们的offset操作有没有错误。
        passage_windows(list[list]),文章对应的经过wordpiece之后的window
        t1_gold (list): 元素为(passage_id,(entity_type,start_idx,end_idx,entity_str))，这里的start_idx和end_idx是在passage中char level的标注
        t1_predict: [(start1,end1),(start2,end2),...]，这里的start和end是在query+window中wordpiece level的标注
        t1_ids(list): 元素为(passage_id,window_id,entity_type)
        window_offset(list): 元素为对应t1_predict window在passage中char level的偏移量
        query_offset(dict): [CLS]+query+[SEP]对应的offset，这里是word piecce level的偏移量（和entity type有关）
    """
    t1_predict1 = []
    for i,(_id, pre) in enumerate(zip(t1_ids,t1_predict)):
        passage_id,window_id,entity_type = _id
        window = tokenizer.tokenize(passage_windows[passage_id][window_id])
        for start,end in pre:
            start1,end1 = start-query_offset[entity_type], end-query_offset[entity_type]
            wordpiece_str = tokenizer.convert_tokens_to_string(window[:start1])
            start2 = wordpice_idx(passage_windows[passage_id][window_id],wordpiece_str)+1
            ent_str = tokenizer.convert_tokens_to_string(window[start1:end1])
            end2 = start2 + wordpice_idx(context[start2:],ent_str)
            start3,end3 = start2+window_offset[i],end2+window_offset[i]
            assert ent_str.lower().replace(" ","")==passages[passage_id][start3:end3].lower().replace(" ","")
            t1_predict1.append((passage_id,(entity_type,start3,end3,passages[passage_id][start3:end3])))
    #print("t1 gold:",t1_gold)
    #print("t1 predict",t1_predict1)
    return get_score(set(t1_gold),set(t1_predict1))


def eval_t2(tokenizer,passages,passage_windows,t2_gold,t2_predict,t2_ids,window_offset,query_offset):
    """
    Args:
        t2_gold: (passage_id,(head_entity,relation_type,end_entity))，其中entity是passage中char leval的四元组(entity_type,start_idx,end_idx,entity_str)
        t2_predict: [(s1,e1),(s1',s2'),...]
        t2_ids: (passage_id,window_id,head_entity,relation_type,end_entity_type)，这里的head entity的索引相对passage而非window的索引
        query_offset(list):[CLS]+query+[SEP]对应的offset，这里是word piecce level的偏移量
    """
    t2_predict1 = []
    for i,(_id, pre) in enumerate(zip(t2_ids,t2_predict)):
        passage_id,window_id,head_entity,relation_type,end_entity_type = _id
        window = tokenizer.tokenize(passage_windows[passage_id][window_id])
        for start, end in pre:
            start1,end1 = start-query_offset[i],end-query_offset[i]
            wordpiece_str = tokenizer.convert_tokens_to_string(window[:start1])
            start2 = wordpice_idx(passage_windows[passage_id][window_id],wordpiece_str)+1
            ent_str = tokenizer.convert_tokens_to_string(window[start1:end1])
            end2 = start2 + wordpice_idx(context[start2:],ent_str)
            start3,end3 = start2+window_offset[i],end2+window_offset[i]
            assert ent_str.lower().replace(" ","")==passages[passage_id][start3:end3].lower().replace(" ","")
            t2_predict1.append((passage_id,(head_entity,relation_type,(end_entity_type,start3,end3,passages[passage_id][start3:end3]))))
    #print("t2 gold:",t2_gold)
    #print("t2 predict:",t2_predict1)
    return get_score(set(t2_gold),set(t2_predict1))


#fixme: eval_t1和eval_t2要改一下,之前对一些函数的接口理解有问题
#todo: 这里为了实现的简便，我们对overlap只考虑union，后续可以考虑求交集
#之前因为考虑overlap的问题，把问题搞复杂了，如果不考虑overlap（即直接union），代码很简单的。
"""
def eval_t1(t1_ids,t1_predict,t1_gold,window_size,overlap,overlap_opt="union"):
    '''
    Args:
        t1_ids (list)：t1_ids[i]为(passage_id,window_id,entity_type)
        t1_predict: list of [(s1,e1), (s2,e2), ...]
        t1_gold: list of (passage_id,entity)
        overlap_opt: 对overlap部分重复预测的处理方式，这里没有实现
    '''
    #首先将预测结果中的在window context中的(s,e)换成passage中的(s,e)
    t1_predict1 = []
    for _id,pre in zip(t1_ids,t1_predict):
        #这里我们对一个window的预测进行修正
        window_id = _id[1]
        offset = (window_size-overlap)*window_id
        ents = []
        for s,e in pre:
            ns,ne = s-offset,e-offset
            ents.append((ns,ne))
        t1_predict1.append((_id[0],_id[1],_id[-1],ents))
    #这里我们考虑按照passage_id进行聚类
    t1_predict2 = []#其元素为(passage_id,[(window_id,entity_type,ents),...])其中ents是(s,e)的列表
    t1_predict2i = []
    cur_passage_id = t1_predict1[0][0]
    for i in range(len(t1_predict1)):
        if t1_predict1[i][0]==cur_passage_id:
            t1_predict2i.append(t1_predict1[i][1:])
        if i+1==len(t1_predict1) or t1_predict1[i+1][0]!=cur_passage_id:
            t1_predict2.append((cur_passage_id,t1_predict2i))
            cur_passage_id = t1_predict1[i+1][0]
            t1_predict2i = []
    t1_predict3 = []#t1其元素为(passage_id,entity_type,[entitiy1,entity2,...])
    for i in range(len(t1_predict2)):
        passage_id,window_pre = t1_predict2[i]
        #这里考虑对窗口overlap的部分取并集，后续可以实验取交集的效果
        entities = {}
        #去掉window_id这个东东。。
        for window_id,ent_type,ents in window_pre:
            entities[ent_type] = entities.get(ent_type,[])+ents
        entities1 = []
        for k,v in entities.items():
            entities1.append((passage_id,k,list(set(v))))
        t1_predict3.extend(entities1)
    #将t1_predict3中的[entity1,entity2...]展开
    t1_predict4 = []
    for passage_id,ent_type,ents in t1_predict3:
        for ent in ents:
            t1_predict4.append((passage_id,(ent_type,)+ent))
    return get_score(set(t1_gold),set(t1_predict4))


def eval_t1(t1_ids,query_offset,t1_predict,t1_gold,window_size,overlap,overlap_opt="union"):
    '''
    Args:
        t1_ids (list)：t1_ids[i]为(passage_id,window_id,entity_type)
        query_offset: 字典，key=实体类型，value=对应的queryoffset
        t1_predict: list of [(s1,e1), (s2,e2), ...]
        t1_gold: list of (passage_id,entity)
        overlap_opt: 对overlap部分重复预测的处理方式，这里没有实现
    '''
    print("t1_ids:",t1_ids)
    print("query_offset:",query_offset)
    print("t1_gold:",t1_gold)
    print("t1_predict:",t1_predict)
    t1_predict1 = []
    for _id, pre in zip(t1_ids,t1_predict):
        passage_id,window_id,ent_type = _id
        offset1 = (window_size-overlap)*window_id#滑窗导致的offset
        offset2 = query_offset[ent_type] #query导致的offset
        for s,e in pre:
            ns,ne = s+offset1-offset2,e+offset1-offset2
            t1_predict1.append((passage_id,(ent_type,ns,ne)))
    print("t1 final predict:",t1_predict1)
    return get_score(set(t1_gold),set(t1_predict1))


def eval_t2(t2_ids, t2_predict,t2_gold,window_size,overlap,overlap_opt="union"):
    '''
    Args:
        t2_gold: (passage_id,(head_entity,relation_type,end_entity))
        t2_predict: [(s1,e1),(s1',s2'),...]
        t2_ids: (passage_id,window_id,head_entity,relation_type,end_entity_type)，这里的s1,e1是passage中的offset
    '''                 
    #修复passage与window的不一致的问题
    t2_predict1 = []
    for _id,pre in zip(t2_ids,t2_predict):
        window_id = _id[1]
        offset = (window_size-overlap)*window_id
        ents = []
        for s,e in pre:
            ns,ne = s-offset,e-offset
            ents.append((ns,ne))
        t2_predict1.append(_id+ents)
    #同样按照passage_id进行聚类
    t2_predict2 = []
    t2_predict2i = []
    cur_passage_id = t2_predict[0][0]
    for i in range(len(t2_predict1)):
        if t2_predict1[i][0]==cur_passage_id:
            t2_predict2i.append(t2_predict2[i][1:])
        if i+1==len(t2_predict2) or t2_predict2[i+1][0]!=cur_passage_id:
            t2_predict2.append((cur_passage_id,t2_predict2i))
            cur_passage_id =  t2_predict2[i+1][0]
            t2_predict2i = []
    t2_predict3 = []#其元素为(passage_id,head_entity,relation_type,[end_entity1,end_entity2,...])
    for i in range(len(t2_predict2)):
        #这里我们去掉window id这个东西
        passage_id, window_pre = t2_predict2[i]
        relations = {}
        for window_id,head_entity,relation_type,end_entity_type,ents in window_pre:
            key = (head_entity,relation_type,end_entity_type)
            relations[key]=ents
        relations1 = []
        for k,v in relations.items():
            relations1.append((passage_id,k,list(set(v))))
        t2_predict3.extend(entities1)
    return get_score(set(t2_gold),set(t2_predict3))

def eval_t2(t2_ids, query_offset,t2_predict,t2_gold,window_size,overlap,overlap_opt="union"):
    '''
    Args:
        t2_gold: (passage_id,(head_entity,relation_type,end_entity))
        query_offset: 列表，代表第i个样本的query的offset。注意这里和eval_t1中的不一样，因为query与实体内容相关了
        t2_predict: [(s1,e1),(s1',s2'),...]
        t2_ids: (passage_id,window_id,head_entity,relation_type,end_entity_type)，注意这里的s1,e1是passage中的offset,不是window中的id
    '''              
    t2_predict1 = []
    for i,(_id, pre) in enumerate(zip(t2_ids,t2_predict)):
        passage_id,window_id,head_entity,relation_type,end_entity_type = _id
        offset1 = (window_size-overlap)*window_id
        offset2 = query_offset[i]
        for s,e in pre:
            ns,ne = s-offset1-offset2,e-offset1-offset2
            t2_predict1.append((passage_id,(head_entity,relation_type,(end_entity_type,ns,ne))))
    return get_score(set(t2_gold),set(t2_predict1))            
"""


def tag_decode(tags,context_mask=None):
    """
    对span的索引取左闭右开
    """
    seq_len = tags.shape[1]
    spans = [[]]*tags.shape[0]
    tags = tags.tolist()
    if not context_mask is None:
        context_mask = context_mask.tolist()
    #确定有答案的样本，以及对应的起点
    has_answer = []
    start_idxs = []
    end_idxs = []
    for i,t in enumerate(tags):
        if t[0]!=tag_idxs['S']:
            has_answer.append(i)
            if context_mask is None:
              mask = [1 if i!=-1 else 0 for i in t]
            else:
              mask = context_mask[i]
            s = mask.index(1,1)
            e = mask.index(0,s)
            start_idxs.append(s)
            end_idxs.append(e)
    for i,s,e in zip(has_answer,start_idxs,end_idxs):
        span = []
        j=s
        while j<e:
            if tags[i][j]==tag_idxs['S']:
                span.append([j,j+1])
                j+=1
            elif tags[i][j]==tag_idxs['B']:
                #不是语法严格的解码，只要在遇到下一个B和S之前找到E就行(前期预测的结果很可能是语法不正确的)
                for k in range(j+1,e):
                    if tags[i][k] in [tag_idxs['B'],tag_idxs['S']]:
                        j=k
                        break
                    elif tags[i][k]==tag_idxs["E"]:
                        span.append([j,k+1])
                        j=k+1
                        break
                if k==e-1 and tags[i][k]==tag_idxs['O']:
                    j=k+1
                    break
                elif k<j+1:
                    j+=1
                    break
            else:
                j+=1
        spans[i]=span
    return spans