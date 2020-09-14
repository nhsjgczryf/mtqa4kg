from tqdm import tqdm
from dataloader import tag_idxs
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

def test_evaluation():
    pass

def predict():
    pass

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
    for i,t in enumerate(tags):
        if t[0]!=tag_idxs['S']:
            has_answer.append(i)
            for j, tj in enumerate(t[1:],1):
                if tj != -1 and (context_mask is None):
                    start_idxs.append(j)
                    break
                if (not context_mask is None) and context_mask[j]!=0:
                    start_idxs.append(j)
                    break
    for i,s in zip(has_answer,start_idxs):
        span = []
        j=s
        while j<seq_len:
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