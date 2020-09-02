"""
对原始数据进行处理，将数据转化为
"""
import os
import argparse
import nltk
parser = argparse.ArgumentParser()
parser.add_argument(" --dataset",choices=['ace2004','ace2005'],type=str)
parser.add_argument(" --path",type=str,help="数据集文件夹的路径")
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


def ace2004_processor(path):
    pass

def ace2005_processor(path):
    """
    对ace2005数据集进行处理
    """
    all_path = [os.path.join(path,t) for t in ['train','dev','test']]
    for p in all_path:
        #对文件进行处理
        with open(p,encoding='utf-8') as f:
            raw_txt =f.read()
            txt = [t for t in raw_txt.split('\n') if t]
        txt1 = txt[3:]#去掉前三行没用的信息
        txt2 = "\n".join(txt1)
        txt3 = nltk.sent_tokenize(txt2)#切分句子
        #得到每


if __name__=="__main__":
    pass