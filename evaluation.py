from tqdm import tqdm
import torch

def get_score(gold_set,predict_set):
    """得到两个集合的precision,recall.f1"""
    TP = len(set.intersection(gold_set,predict_set))
    precision = TP/(len(predict_set)+1e-6)
    recall = TP/(len(gold_set)+1e-6)
    f1 = 2*precision*recall/(precision+recall+1e-6)
    return precision,recall,f1

def evaluation(model,dataloader):
    if hasattr(model,'module'):
        model = model.module
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()
    gold = []
    predict = []
    tqdm_dataloader = tqdm(dataloader,desc="eval")
    with torch.no_grad():
        for batch in tqdm_dataloader:
