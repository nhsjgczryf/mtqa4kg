"""
对原始数据进行处理，将数据转化为
"""
import os
import argparse
import json
import nltk


ace2004_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2004_relations = ['ART', 'EMP-ORG', 'GPE-AFF', 'OTHER-AFF', 'PER-SOC', 'PHYS']

ace2005_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2005_relations = ['ART', 'GEN-AFF', 'ORG-AFF', 'PART-WHOLE', 'PER-SOC', 'PHYS']

#这个后续可以优化，假设所有得组合都有可能
ace2004_relation_triples = [(ent1,rela,ent2) for rela in ace2004_relations for ent1 in ace2004_entities for ent2 in ace2004_entities]
ace2005_relation_triples = [(ent1,rela,ent2) for rela in ace2005_relations for ent1 in ace2005_entities for ent2 in ace2005_entities]

parser = argparse.ArgumentParser()
parser.add_argument(" --dataset_tag", choices=['ace2004', 'ace2005'], type=str)
parser.add_argument(" --dataset_path", type=str, help="数据集文件夹的路径")
parser.add_argument(" --query_template_path", type=str, help="query模板")
parser.add_argument(" --outputpath", type=str)
args = parser.parse_args()

with open(args.query_template_path, encoding='utf-8') as f:
    question_templates = json.load(f)
entities = ace2004_entities if args.dataset_tag == 'ace2004' else ace2005_entities
relations = ace2004_relations if args.dataset_tag == 'ace2004' else ace2005_relations
relation_triples = ace2004_relation_triples if args.dataset_tag == 'ace2004' else ace2004_relation_triples

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

def get_sent_er(txt,entities,relations):
    """
    得到句子级别的标注
    Args:
        txt: 待处理的文本
        entities: 对应的实体标注，是四元组(entity_type, start_idx, end_idx,entity)的列表,不包含end_idx的内容
        relations: 对应的关系标注,(relation_type,entity1,entity2)三元组的列表
    Returns:
        句子级别的标注,list of [句子，实体列表，关系列表]
    """
    sent = nltk.sent_tokenize(txt)
    sent_idx = [0]
    for i,s in enumerate(sent):
        sent_idx.append(sent_idx[-1]+len(s)+i)
    sent_range = []#每个句子在txt中对应的索引
    for i in range(1,len(sent_idx)):
        sent_range.append((sent_idx[i-1],sent_idx[i]-1))
    ser = []#元素为[句子，实体列表，关系列表]的列表
    e_dict = {}#用来记录某个实体对应在哪个句子
    for i,s,e in enumerate(sent_range):
        es = []#实体集合
        for j,(entity_type, start_idx, end_idx,entity_str) in enumerate(entities):
            if start_idx>=s and end_idx<=e:
                es.append((entity_type,start_idx-s,end_idx-s,entity_str))
                e_dict[j]=i
                assert sent[i][es[1]:es[2]]==entity_str
        ser.append([sent[i],es])
    for r,e1,e2 in relations:
        i1,i2 = e_dict[e1],e_dict[e2]
        assert i1==i2
        t1,s1,e1,es1 = entities[e1][0],entities[e1][1]-sent_range[i1][0],entities[e1][2]-sent_range[i1][0],entities[e1][3]
        t2,s2,e2,es2 = entities[e2][0],entities[e2][1]-sent_range[i1][0],entities[e2][2]-sent_range[i1][0],entities[e2][3]
        ser[i1].append((r,(t1,s1,e1,es1),(t2,s2,e2,es2)))
    return ser


def get_question(head_entity,relation=None,end_entity=None):
    """
    Args:
        head_entity: (entity_type,start_idx,end_idx,entity_string) or entity_type
        relation: (relation_type,start_entity,end_entity)
        end_entity:(entity_type,start_idx,end_idx,entity_string) or entity_type
    """
    if relation==None:
        question = question_templates['qa_turn1'][head_entity[0]] if  isinstance(head_entity,tuple) else question_templates['qa_turn1'][head_entity]
    else:
        end_entity = end_entity[0] if isinstance(end_entity,tuple) else end_entity
        question = question_templates['qa_turn2'][(head_entity[0],relation[0],end_entity)]
        question = question.replace('XXX',head_entity[3])
    return question


def sent2qas(ser,allow_impossible=False):
    sent, ents, relas = ser
    res = {'context':sent}
    if not allow_impossible:
        #构造第一轮问答
        qat1 = {}
        for en in ents:
            question = get_question(en)
            qat1[question] = qat1.get(question,[])+[en]
        #构造第二轮问答
        qat2 = {}
        for rel in relas:
            rel_type,head_ent,end_ent = rel
            question = get_question(head_ent,rel_type,end_ent)
            qat2[question] = qat2.get(question,[])+[rel]
    else:
        #构造一轮问答
        dict1 = {k:get_question(k) for k in entities}
        qat1 = {k:[] for k in dict1}
        for en in ents:
            q = dict1[en[0]]
            qat1[q].append(en)
        #构造第二轮问答
        dict2 = {}
        for ent in ents:
            for rel_type in relations:
                for ent_type in entities:
                    k = (ent[0],rel_type,ent_type)
                    if k in relation_triples:
                        dict2[k]=get_question(*k)
        qat2 = {k:[] for k in dict2}
        for rel in relas:
            head_ent, rela_type, end_ent = rel
            k = (head_ent[0],rela_type,end_ent[0])
            qat2[k].append(rel)
    qas = [qat1,qat2]
    res["qa_pairs"]=qas
    return res


def process(dataset_path):
    all_path = [os.path.join(dataset_path,t) for t in ['train','dev','test']]
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
            #开始构造数据集
            for ser in sent_er:
                data.append(sent2qas(ser))
        save_path = os.path.join(args.outputpath,os.path.split(p)[-1])
        with open(save_path,encoding='utf-8') as f:
            json.dump(data,f)

if __name__=="__main__":
