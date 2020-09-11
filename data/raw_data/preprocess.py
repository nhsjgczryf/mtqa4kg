"""
对原始数据进行处理，将数据转化为
"""
import os
import argparse
import json
import nltk

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_tag",default='ace2005', choices=['ace2004', 'ace2005'],  type=str)
parser.add_argument("--dataset_dir", type=str,default=r'C:\Users\hkcs\Desktop\mtqa4kg\data\raw_data\ACE2005', help="数据集文件夹的路径")
parser.add_argument("--query_template_path",default=r"C:\Users\hkcs\Desktop\mtqa4kg\data\query_templates\ace2005.json", type=str, help="query模板")
parser.add_argument("--allow_impossible",action="store_true")
parser.add_argument("--window_size",type=int,default=100)
parser.add_argument("--overlap",type=int,default=50)
parser.add_argument("--output_dir", default=r"C:\Users\hkcs\Desktop\mtqa4kg\data\cleaned_data\ACE2005",type=str)
args = parser.parse_args()


ace2004_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2004_relations = ['ART', 'EMP-ORG', 'GPE-AFF', 'OTHER-AFF', 'PER-SOC', 'PHYS']

ace2005_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2005_entities_full = ["facility","geo political","location","organization","person","vehicle","weapon"]
ace2005_relations = ['ART', 'GEN-AFF', 'ORG-AFF', 'PART-WHOLE', 'PER-SOC', 'PHYS']
ace2005_relations_full = ["artifact","gen affilliation",'organization affiliation','part whole','person social','physical']

#这个后续可以优化，假设所有得组合都有可能
ace2004_relation_triples = [(ent1,rela,ent2) for rela in ace2004_relations for ent1 in ace2004_entities for ent2 in ace2004_entities]
#ace2005_relation_triples = [(ent1,rela,ent2) for rela in ace2005_relations for ent1 in ace2005_entities for ent2 in ace2005_entities]

ace2005_relation_st = {
    'PHYS':{
        'ARG1':[
            ['PER'],
            ['PER','FAC','GPE','LOC']
        ],
        'ARG2':[
            ['FAC','LOC','GPE'],
            ['FAC','GPE','LOC']
        ]
    },
    'PART-WHOLE':{
        'ARG1':[
            ['FAC','LOC','GPE'],
            ['ORG'],
            ['VEH'],
            ['WEA']
        ],
        'ARG2':[
            ['FAC','LOC','GPE'],
            ['ORG','GPE'],
            ['VEH'],
            ['WEA']
        ]
    },
    'PER-SOC':{
        'ARG1':[
            ['PER']
        ],
        'ARG2':[
            ['PER']
        ]
    },
    'ORG-AFF':{
        'ARG1':[
            ['PER','ORG','GPE']
        ],
        'ARG2':[
            ['ORG','GPE'],
        ]
    },
    'ART':{
        'ARG1':[
            ['PER', 'ORG', 'GPE']
        ],
        'ARG2':[
            ['WEA','VEH','FAC']
        ]
    },
    'GEN-AFF':{
        'ARG1':[
            ['PER'],
            ['ORG']
        ],
        'ARG2':[
            ['PER','LOC','GPE','ORG'],
            ['LOC','GPE']
        ]
    }
}
ace2005_relation_triples = []
for rel,val in ace2005_relation_st.items():
    arg1 = val['ARG1']
    arg2 = val['ARG2']
    for a1s,a2s in zip(arg1,arg2):
        for a1 in a1s:
            for a2 in a2s:
                if (a1,rel,a2) not in ace2005_relation_triples:
                    ace2005_relation_triples.append((a1,rel,a2))

with open(args.query_template_path, encoding='utf-8') as f:
    question_templates = json.load(f)
entities = ace2004_entities if args.dataset_tag == 'ace2004' else ace2005_entities
relations = ace2004_relations if args.dataset_tag == 'ace2004' else ace2005_relations
relation_triples = ace2004_relation_triples if args.dataset_tag == 'ace2004' else ace2005_relation_triples

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
            entities.append([al[1],int(al[2])+offset,int(al[3])+offset,al[4]])
        else:
            relations.append([al[1],al[2][5:],al[3][5:]])
    return entities,relations

def aligment_ann(original, newtext, ann_file, offset):
    annotation = []
    terms = {}
    ends = {}
    for line in open(ann_file):
        if line.startswith('T'):
            annots = line.rstrip().split("\t", 2)
            typeregion = annots[1].split(" ")
            start = int(typeregion[1]) - offset
            end = int(typeregion[2]) - offset
            if not start in terms:
                terms[start] = []
            if not end in ends:
                ends[end] = []
            if len(annots) == 3:
                terms[start].append([start, end, annots[0], typeregion[0], annots[2]])
            else:
                terms[start].append([start, end, annots[0], typeregion[0], ""])
            ends[end].append(start)
        else:
            annotation.append(line)
    orgidx = 0
    newidx = 0
    orglen = len(original)
    newlen = len(newtext)

    while orgidx < orglen and newidx < newlen:
        if original[orgidx] == newtext[newidx]:
            orgidx += 1
            newidx += 1
        elif newtext[newidx] == '\n':
            newidx += 1
        elif original[orgidx] == '\n':
            orgidx += 1
        elif newtext[newidx] == ' ':
            newidx += 1
        elif original[orgidx] == ' ':
            orgidx += 1
        elif newtext[newidx] == '\t':
            newidx += 1
        elif original[orgidx] == '\t':
            orgidx += 1
        elif newtext[newidx] == '.':
            # ignore extra "." for stanford
            newidx += 1
        else:
            assert False, "%d\t$%s$\t$%s$" % (orgidx, original[orgidx:orgidx + 20], newtext[newidx:newidx + 20])
        if orgidx in terms:
            for l in terms[orgidx]:
                l[0] = newidx
        if orgidx in ends:
            for start in ends[orgidx]:
                for l in terms[start]:
                    if l[1] == orgidx:
                        l[1] = newidx
            del ends[orgidx]
    entities = []
    relations = []
    dict1 = {}
    i=0
    for ts in terms.values():
        for term in ts:
            if term[4]=="":
                entities.append([term[2], term[3], term[0], term[1], newtext[term[0]:term[1]]])
            else:
                assert newtext[term[0]:term[1]].replace("&AMP;", "&").replace("&amp;", "&").replace(" ", "").replace(
                    "\n", "") == term[4].replace(" ", ""), newtext[term[0]:term[1]] + "<=>" + term[4]
                entities.append([term[2], term[3], term[0], term[1], newtext[term[0]:term[1]].replace("\n", " ")])
            dict1[term[2]] = i
            i += 1
    for rel in annotation:
        rel_id,rel_type,rel_e1,rel_e2 = rel.strip().split()
        rel_e1 = rel_e1[5:]
        rel_e2 = rel_e2[5:]
        relations.append([rel_id,rel_type,rel_e1,rel_e2])
    relations1 = []
    for rel in relations:
        _,rel_type,rel_e1,rel_e2=rel
        rel_e1_idx = dict1[rel_e1]
        rel_e2_idx = dict1[rel_e2]
        relations1.append([rel_type,rel_e1_idx,rel_e2_idx])
    entities1 = [[ent[1],ent[2],ent[3],ent[4]] for ent in entities]
    return entities1,relations1

def chunk_passage(txt,window_size,overlap):
    """
    对txt进行滑窗切块处理，返回被切分后的文本块，以及对应的文本块在原txt中的起止位置，
    注意这里的滑窗以空白符分隔后的token为基本单位
    """
    words = txt.split()
    assert " ".join(words)==txt
    chunks = []
    regions = []
    for i in range(0,len(words),window_size-overlap):
        c = words[i:i+window_size]
        c1 = " ".join(c)
        chunks.append(c1)
        c_start = len(" ".join(words[:i]))+1 if i!=0 else 0
        c_end = c_start+len(c1)
        regions.append((c_start,c_end))
    return chunks,regions


def get_sent_er(txt,entities,relations,window_size,overlap):
    """
    得到句子级别的标注
    Args:
        txt: 待处理的文本
        entities: 对应的实体标注，是四元组(entity_type, start_idx, end_idx,entity)的列表,不包含end_idx的内容
        relations: 对应的关系标注,(relation_type,entity1_idx,entity2_idx)三元组的列表，其中entity_idx是对应的entity在entities列表中的
    Returns:
        句子级别的标注,list of [句子，实体列表，关系列表]
    """
    #sent = nltk.sent_tokenize(txt)
    #sent_idx = [0]
    #for i,s in enumerate(sent):
    #    sent_idx.append(sent_idx[-1]+len(s)+1)
    #sent_range = []#每个句子在txt中对应的索引
    #for i in range(1,len(sent_idx)):
    #    sent_range.append((sent_idx[i-1],sent_idx[i]-1))
    sent,sent_range = chunk_passage(txt,window_size,overlap)
    ser = [["",[],[]] for i in range(len(sent_range))]#元素为[句子，实体列表，关系列表]的列表
    e_dict = {}#用来记录某个实体对应在哪个句子
    for i,(s,e) in enumerate(sent_range):
        es = []#实体集合
        for j,(entity_type, start_idx, end_idx,entity_str) in enumerate(entities):
            if start_idx>=s and end_idx<=e:
                nstart_idx,nend_idx = start_idx-s,end_idx-s
                if sent[i][nstart_idx:nend_idx]==entity_str:
                    es.append((entity_type,nstart_idx,nend_idx,entity_str))
                    e_dict[j]=e_dict.get(j,[])+[i]
                else:
                    print("实体和对应的索引不一致！")
        ser[i][0]=sent[i]
        ser[i][1].extend(es)
    for r,e1i,e2i in relations:
        if e1i not in e_dict or e2i not in e_dict:
            print("实体丢失引起关系出错！")
            continue
        i1s,i2s = e_dict[e1i],e_dict[e2i]
        intersec = set.intersection(set(i1s),set(i2s))
        if intersec:
            for i in intersec:
                t1,s1,e1,es1 = entities[e1i][0],entities[e1i][1]-sent_range[i][0],entities[e1i][2]-sent_range[i][0],entities[e1i][3]
                t2,s2,e2,es2 = entities[e2i][0],entities[e2i][1]-sent_range[i][0],entities[e2i][2]-sent_range[i][0],entities[e2i][3]
                ser[i][2].append((r,(t1,s1,e1,es1),(t2,s2,e2,es2)))
        else:
            print("关系的两个实体不在一个句子上")
    return ser

def get_question(head_entity,relation_type=None,end_entity_type=None):
    """
    Args:
        head_entity: (entity_type,start_idx,end_idx,entity_string) or entity_type
    """
    if relation_type==None:
        question = question_templates['qa_turn1'][head_entity[0]] if  isinstance(head_entity,tuple) else question_templates['qa_turn1'][head_entity]
    else:
        question = question_templates['qa_turn2'][str((head_entity[0],relation_type,end_entity_type))]
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
            question = get_question(head_ent,rel_type,end_ent[0])
            qat2[question] = qat2.get(question,[])+[rel]
    else:
        #构造一轮问答
        dict1 = {k:get_question(k) for k in entities}
        qat1 = {dict1[k]:[] for k in dict1}
        for en in ents:
            q = dict1[en[0]]
            qat1[q].append(en)
        #构造第二轮问答
        dict2 = {}
        for ent in ents:
            for rel_type in relations:
                for ent_type in entities:
                    tk = (ent[0],rel_type,ent_type)
                    if tk in relation_triples:
                        k = (ent,rel_type,ent_type)
                        dict2[k]=get_question(*k)
        qat2 = {dict2[k]:[] for k in dict2}
        for rel in relas:
            rela_type,head_ent,end_ent = rel
            k = (head_ent,rela_type,end_ent[0])
            try:
                q = dict2[k]
                qat2[q].append(rel)
            except:
                print("似乎出现了未定义的关系：",rel)
    qas = [qat1,qat2]
    res["qa_pairs"]=qas
    return res


def process(dataset_dir,allow_impossible=False,window_size=100,overlap=50):
    all_path = [os.path.join(dataset_dir, t) for t in ['train', 'dev', 'test']]
    for p in all_path:
        #对文件进行处理
        ann_files = []
        txt_files = []
        data = []
        for i in os.listdir(p):
            if i.endswith('txt'):
                txt_files.append(os.path.join(p,i))
            else:
                ann_files.append(os.path.join(p,i))
        for ann_path,txt_path in zip(ann_files,txt_files):
            with open(txt_path,encoding='utf-8') as f:
                raw_txt =f.read()
                txt = [t for t in raw_txt.split('\n') if t.strip()]
            #去掉前三行无用的信息
            ntxt = "\n".join(txt[3:])
            #去掉多余的空白符
            ntxt = " ".join(ntxt.split())
            #得到偏移量
            offset = raw_txt.index(txt[3])# 得到第一个句子的偏移
            #解析得到实体和关系
            #entities, relations = parse_ann(ann,offset)
            entities,relations = aligment_ann(raw_txt[offset:],ntxt,ann_path,offset)
            #得到每个句子里面的实体以及关系
            sent_er = get_sent_er(ntxt,entities,relations,window_size,overlap)
            #开始构造数据集
            for ser in sent_er:
                data.append(sent2qas(ser,allow_impossible))
        save_path = os.path.join(args.output_dir,os.path.split(p)[-1]+".json")
        with open(save_path,'w',encoding='utf-8') as f:
            json.dump(data,f)

if __name__=="__main__":
    #process(args.dataset_dir)
    process(args.dataset_dir,allow_impossible=True)