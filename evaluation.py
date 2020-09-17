from tqdm import tqdm
from dataloader import tag_idxs,load_t2_data
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
    model.eval()
    t1_predict = []
    t2_predict = []
    #第一轮问答
    with torch.no_grad():
        for i,batch in enumerate(tqdm(t1_dataloader,desc="t1")):
            txt_ids,attention_mask,token_type_ids = batch['txt_ids'],batch['attention_mask'],batch['token_type_ids']
            tag_idxs = model(txt_ids.to(device),attention_mask.to(device),token_type_ids.to(device))
            predict_spans = tag_decode(tag_idxs,context_mask)
            t1_predict.extend(predict_spans)
    #进行第二轮问答
    t2_dataloader = load_t2_data(t1_dataloader.dataset,t1_predict)
    with torch.no_grad():
        for i,batch in enumerate(tqdm(t2_dataloader,desc="t2")):
            txt_ids,attention_mask,token_type_ids = batch['txt_ids'],batch['attention_mask'],batch['token_type_ids']
            tag_idxs = model(txt_ids.to(device),attention_mask.to(device),token_type_ids.to(device))
            predict_spans = tag_decode(tag_idxs,context_mask)
            t2_predict.extend(predict_spans)
    #获取一些需要需要的信息
    t1_ids = t1_dataloader.dataset.t1_ids
    t2_ids = t2_dataloader.dataset.t2_ids
    window_size = t1_dataloader.dataset.window_size
    overlap = t1_dataloader.dataset.overlap
    t1_gold = t1_dataloader.dataset.t1_gold
    t2_gold = t2_dataloader.dataset.t2_gold
    #第一阶段的评估，即评估我们的ner的结果
    p1,r1,f1 = eval_t1(t1_ids,t1_predict,t1_gold,window_size,overlap)
    #第二阶段的评估，即评估我们的ner+re的综合结果
    p2,r2,f2 = eval_t2(t2_ids, t2_predict,t2_gold,window_size,overlap)
    return (p1,r1,f1),(p2,r2,f2)


#fixme: eval_t1和eval_t2要改一下,之前对一些函数的接口理解有问题
#todo: 这里为了实现的简便，我们对overlap只考虑union，后续可以考虑求交集
def eval_t1(t1_ids,t1_predict,t1_gold,window_size,overlap,overlap_opt="union"):
    """
    Args:
        t1_ids (list)：t1_ids[i]为(passage_id,window_id,query_id/entity_type_id)
        t1_predict: list of [(s1,e1), (s2,e2), ...]
        t1_gold: list of (passage_id,entity_type_id,[(s1,e1),(s2,e2),...])
        overlap_opt: 对overlap部分重复预测的处理方式，这里没有实现
    """
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
    t1_predict2 = []#其元素为(passage_id,[(window_id,entity_type_id,ents),...])其中ents是(s,e)的列表
    t1_predict2i = []
    cur_passage_id = t1_predict1[0][0]
    for i in range(len(t1_predict1)):
        if t1_predict1[i][0]==cur_passage_id:
            t1_predict2i.append(t1_predict1[i][:1])
        if i+1==len(t1_predict1) or t1_predict1[i+1][0]!=cur_passage_id:
            t1_predict2.append((cur_passage_id,t1_predict2i))
            cur_passage_id = t1_predict1[i+1][0]
            t1_predict2i = []
    t1_predict3 = []
    for i in range(len(t1_predict2)):
        passage_id,window_pre = t1_predict2[i]
        #这里考虑对窗口overlap的部分取并集，后续可以实验取交集的效果
        entities = {}
        #去掉window_id这个东东。。
        for window_id,ent_type_id,ents in window_pre:
            entities[ent_type_id] = entities.get(ent_type_id,[])+ents
        entities1 = []
        for k,v in entities.items():
            entities1.append((passage_id,k,list(set(v)))
        t1_predict3.extend(entities1)
    return get_score(set(t1_gold),set(t1_predict3))


def eval_t2(t2_ids, t2_predict,t2_gold,window_size,overlap,overlap_opt="union"):
    """
    Args:
        t2_gold: (passage_id,[(relation_type_id,(s1,e1),(s2,e2)),(relation_type_id`,(s1`,e1`),(s2`,e2)`)])
        t2_predict: [(s1,e1),(s1',s2'),...]
        t2_ids: (passage_id,windos_id,(s1,e1),relation_type_id)，这里的s1,e1是window中的offset
    """
    #这里我们同样有一个fix offset的操作
    #t2_predict1 = []
    #for _id, pre in zip(t2_ids,t2_predict):
    #    window_id = _id[1]
    #    offset = (window_size-overlap)*window_id
    #    ns1,ne1 = _id[2][0]-offset,_id[2][0]-offset
    #    ns2,ne2 = pre[0]-offset,pre[1]-offset
    #    t2_predict1.append((_id[0],(ns1,ne1),_id[-1],(ns2,ne2)))
    #return get_score(set(t2_gold),set(t2_predict1))


def tag_decode(tags,context_mask=None):
    """
    对span的索引取两侧闭区间
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
                span.append([j,j])
                j+=1
            elif tags[i][j]==tag_idxs['B']:
                #不是语法严格的解码，只要在遇到下一个B和S之前找到E就行
                for k in range(j+1,seq_len):
                    if tags[i][j] in [tag_idxs['B'],tag_idxs['S']]:
                        j=k
                        break
                    if tags[i][j]==tag_idxs["E"]:
                        span.append([j,k])
                        j=k+1
                        break
            else:
                j+=1
        spans[i]=span
    return spans