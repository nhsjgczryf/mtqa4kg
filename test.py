import pickle
import os
import torch
from evaluation import test_evaluation,dev_evaluation,full_dev_evaluation
from model import MyModel
from dataloader import load_t1_data,load_data

pretrained_model_path = "/home/wangnan/pretrained_models/bert-base-uncased"
#file = 'AFP_ENG_20030305'
#test_path = "/home/wangnan/mtqa4kg/data/cleaned_data/ACE2005/bert-base-uncased_overlap_0_window_300_threshold_4_max_distance_-1_test_onep/AFP_ENG_20030305_test.json"
#dev_path = "/home/wangnan/mtqa4kg/data/cleaned_data/ACE2005/bert-base-uncased_overlap_0_window_300_threshold_4_max_distance_-1_test_onep/AFP_ENG_20030305_train.json"
test_path = '/home/wangnan/mtqa4kg/data/cleaned_data/ACE2005/test.json'
#dev_path = "/home/wangnan/mtqa4kg/data/cleaned_data/ACE2005/bert-base-uncased_overlap_0_window_300_threshold_4_max_distance_-1_test_onep/test_train.json"
test_batch=10
dev_batch=10
model_dir = '/home/wangnan/mtqa4kg/checkpoints/ACE2005/2020_10_06_12_54_04/'
file = "checkpoint_4.cpt"
args = pickle.load(open(model_dir+'args',"rb"))
checkpoint = torch.load(model_dir+file,map_location=torch.device("cpu"))

model_state_dict = checkpoint['model_state_dict']
optimizer_state_dict = checkpoint['optimizer_state_dict']
scheduler_state_dict = checkpoint['scheduler_state_dict']

mymodel = MyModel(args)
mymodel.load_state_dict(model_state_dict,strict=False)
device = torch.device("cuda") if  torch.cuda.is_available() else torch.device("cpu")
mymodel.to(device)


test_dataloader = load_t1_data(test_path,args.pretrained_model_path,300,0,test_batch,args.max_len)
(p1,r1,f1),(p2,r2,f2) = test_evaluation(mymodel,test_dataloader,5,-1,False,True)

print("Turn 1: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p1,r1,f1))
print("Turn 2: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p2,r2,f2))
print('\n')

#dev_dataloader = load_data(dev_path,dev_batch,args.max_len,pretrained_model_path,False,False,1e-10,threshold=4)
#p,r,f = full_dev_evaluation(mymodel,dev_dataloader)
#print("dev: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p,r,f))



