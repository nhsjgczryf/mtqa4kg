import pickle
import os
import torch
from evaluation import test_evaluation,dev_evaluation,full_dev_evaluation,
from model import MyModel
from dataloader import load_t1_data,load_data


test_path = "/home/wangnan/mtqa4kg/data/cleaned_data/ACE2005/bert_base_uncased/test.json"
test_batch=5
model_dir = '/home/wangnan/mtqa4kg/checkpoints/ACE2005/2020_09_28_11_50_20/'
file = "checkpoint_4.cpt"
args = pickle.load(open(model_dir+'args',"rb"))
checkpoint = torch.load(model_dir+file,map_location=torch.device("cpu"))

model_state_dict = checkpoint['model_state_dict']
optimizer_state_dict = checkpoint['optimizer_state_dict']
scheduler_state_dict = checkpoint['scheduler_state_dict']

mymodel = MyModel(args)
mymodel.load_state_dict(model_state_dict)
device = torch.device("cuda") if  torch.cuda.is_available() else torch.device("cpu")
mymodel.to(device)

test_dataloader = load_t1_data(test_path,args.pretrained_model_path,args.window_size,args.overlap,test_batch,args.max_len)

#test_dataloader = load_t1_data(test_path,args.pretrained_model_path,args.window_size,args.overlap,test_batch,args.max_len)
(p1,r1,f1),(p2,r2,f2) = test_evaluation(mymodel,test_dataloader)

print("Turn 1: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p1,r1,f1))
print("Turn 2: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p2,r2,f2))


#dev_dataloader = load_data(dev_path,dev_batch,args.max_len,args.pretrained_model_path)
#dev_dataloader = pickle.load(open('/home/wangnan/mtqa4kg/data/cleaned_data/ACE2005/dev_4_300_bert-base-uncased',"rb"))
#p,r,f = dev_evaluation(mymodel,dev_dataloader)
#print("dev: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p,r,f))



