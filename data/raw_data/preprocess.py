"""
对原始数据进行处理，将数据转化为
"""
import os
import argparse
import nltk
parser = argparse.ArgumentParser()
parser.add_argument(" --dataset_tag",choices=['ace2004','ace2005'],type=str)
parser.add_argument(" --dataset_path",type=str,help="数据集文件夹的路径")
parser.add_argument(" --query_template",type=str,help="query模板")
parser.add_argument(" --outputpath",type=str)
args = parser.parse_args()

def get_ann(origin_txt,origin_ann,new_txt):
    """
    得到新的标注结果,即返回所有的实体和关系
    Args:
        origin_txt: 没有经过去除前三行的
    """
    return entites,relations


def get_sent_er(txt,entities,relations):
    """
    得到句子级别的标注
    Args:
        txt: 待处理的文本
        ann: 对应的标注
    Returns:
        句子级别的标注
    """
    sent = nltk.sent_tokenize(txt)
    return None

def parse_ann(ann,offset=0):
    """对.ann文件解析"""
    ann_list = ann.split('\n')
    ann_list = [an.split('\t') for an in ann_list if an]
    for i, ann in enumerate(ann_list):
        ann_list[i][1]=ann_list[i][1].split()
        empty = []
        for ai in ann_list[i]:
            empty.extend(ai) if isinstance(ai,list) else empty.append(ai)
        ann_list[i] = empty
    entities = []
    relations = []
    for al in ann_list:
        if al[0][0]=='T':
            entities.append([al[0],al[1],int(al[2])+offset,int(al[3])+offset,al[4]])
        else:
            relations.append([al[0],al[1],al[2][5:],al[3][5:]])
    return entities,relations

def ace2004_processor(path):
    pass

def ace2005_processor(path):
    """
    对ace2005数据集进行处理
    """
    all_path = [os.path.join(path,t) for t in ['train','dev','test']]
    for p in all_path:
        #对文件进行处理
        ann_files = []
        txt_files = []
        data = []
        for i in os.listdir(p):
            if i.endswith('txt'):
                txt_files.append(i)
            else:
                ann_files.append(i)
        for ann_path,txt_path in zip(ann_files,txt_files):
            with open(txt_path,encoding='utf-8') as f:
                raw_txt =f.read()
                txt = [t for t in raw_txt.split('\n') if t.strip()]
            with open(ann_path,encoding='utf-8') as f:
                ann = f.read()
            #去掉前三行无用的信息
            ntxt = "\n".join(txt[3:])
            #得到偏移量
            offset = raw_txt.index(txt[3])# 得到第一个句子的偏移
            #解析得到实体和关系
            entities, relations = parse_ann(ann,offset)
            #得到每个句子里面的实体以及关系
            sent_er = get_sent_er(ntxt,entities,relations)


if __name__=="__main__":
    pass