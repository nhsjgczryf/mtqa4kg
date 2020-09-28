"""
对原始数据进行处理，将数据转化为
"""
import os
import json
import re
from tqdm import tqdm
from contants import *

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
    #保证uncased的tokenizer也可以对齐
    original = original.lower()
    newtext = newtext.lower()
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
                #assert newtext[term[0]:term[1]].replace("&AMP;", "&").replace("&amp;", "&").replace(" ", "").replace(
                #    "\n", "") == term[4].replace(" ", "").lower(), newtext[term[0]:term[1]] + "<=>" + term[4]
                assert newtext[term[0]:term[1]].replace(" ","").replace('\n',"").replace("&AMP;", "&").replace("&amp;", "&")== \
                       term[4].replace(" ", "").lower(), newtext[term[0]:term[1]] + "<=>" + term[4]
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

def passage_blocks(txt,window_size,overlap):
    """
    txt是wordpiece的列表
    对txt进行滑窗切块处理，返回被切分后的文本块，以及对应的文本块在原txt中的起止位置，
    """
    blocks = []
    regions = []
    for i in range(0,len(txt),window_size-overlap):
        b = txt[i:i+window_size]
        blocks.append(b)
        regions.append((i,i+window_size))
    return blocks,regions

def get_block_er(txt,entities,relations,window_size,overlap):
    """
    得到block级别的标注
    Args:
        txt: 待处理的文本
        entities: 对应的实体标注，是四元组(entity_type, start, end,entity)的列表,不包含end_idx的内容
        relations: 对应的关系标注,(relation_type,entity1_idx,entity2_idx)三元组的列表，其中entity_idx是对应的entity在entities列表中的
    Returns:
        block级别的标注,list of [block，实体列表，关系列表]
    """
    blocks, block_range = passage_blocks(txt,window_size,overlap)
    ber = [[[],[],[]] for i in range(len(block_range))]
    e_dict = {}#用来记录实体在哪个block
    for i,(s,e) in enumerate(block_range):
        es =[]
        for j,(entity_type, start, end,entity_str) in enumerate(entities):
            if start>=s and end<=e:
                nstart,nend = start-s,end-s
                if tokenizer.convert_tokens_to_string(blocks[i][nstart:nend])==entity_str:
                    es.append((entity_type,nstart,nend,entity_str))
                    e_dict[j]=e_dict.get(j,[])+[i]
                else:
                    print("实体和对应的索引不一致！")
        ber[i][0]=blocks[i]
        ber[i][1].extend(es)
    for r,e1i,e2i in relations:
        if e1i not in e_dict or e2i not in e_dict:
            print("实体丢失引起关系出错！")
            continue
        i1s,i2s = e_dict[e1i],e_dict[e2i]
        intersec = set.intersection(set(i1s),set(i2s))
        if intersec:
            for i in intersec:
                t1,s1,e1,es1 = entities[e1i][0],entities[e1i][1]-block_range[i][0],entities[e1i][2]-block_range[i][0],entities[e1i][3]
                t2,s2,e2,es2 = entities[e2i][0],entities[e2i][1]-block_range[i][0],entities[e2i][2]-block_range[i][0],entities[e2i][3]
                ber[i][2].append((r,(t1,s1,e1,es1),(t2,s2,e2,es2)))
        else:
            print("关系的两个实体不在一个句子上")
    return ber


def get_question(question_templates,head_entity,relation_type=None,end_entity_type=None):
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

def block2qas(ber,dataset_tag,question_templates,title=''):
    if dataset_tag.lower()=="ace2004":
        entities = ace2004_entities
        relations = ace2004_relations
    elif dataset_tag.lower()=='ace2005':
        entities = ace2005_entities 
        relations = ace2005_relations
    else:
        raise Exception("不支持的数据集")
    block,ents,relas=ber
    res = {'context':block,'title':title}
    # 构造第一轮问答
    dict1 = {k: get_question(question_templates,k) for k in entities}
    qat1 = {dict1[k]: [] for k in dict1}
    for en in ents:
        q = dict1[en[0]]
        qat1[q].append(en)
    #构造第二轮问答
    dict2 = {}
    for ent in ents:
        for rel_type in relations:
            for ent_type in entities:
                # 这里我们考虑所有的关系可能
                k = (ent, rel_type, ent_type)
                dict2[k] = get_question(question_templates,ent, rel_type, ent_type)
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


def char_to_wordpiece(passage,entities,tokenizer):
    """返回wordpiece后的passage,以及对应的entities标注"""
    entities1 = []
    tpassage = tokenizer.tokenize(passage)
    for ent in entities:
        ent_type,start,end,ent_str = ent
        s = tokenizer.tokenize(passage[:start])
        start1 = len(s)
        ent_str1 = tokenizer.tokenize(ent_str)
        end1 = start1 + len(ent_str1)#这里取的右侧开区间
        ent_str2 = tokenizer.convert_tokens_to_string(ent_str1)
        assert tpassage[start1:end1]==ent_str1
        entities1.append((ent_type,start1,end1,ent_str2))
    return entities1

def process(data_dir,output_dir,tokenizer,is_test,window_size,overlap,dataset_tag,question_templates):
    """
    output_dir的名称最好包含tokenizer的信息，因为这里不同的tokenizer得到的数据是不一样的
    """
    ann_files = []
    txt_files = []
    data = []
    for f in os.listdir(data_dir):
        if f.endswith('.txt'):
            txt_files.append(os.path.join(data_dir,f))
        elif f.endswith('.ann'):
            ann_files.append(os.path.join(data_dir,f))
    for ann_path,txt_path in tqdm(zip(ann_files,txt_files),total=len(ann_files)):
        with open(txt_path,encoding='utf-8') as f:
            raw_txt = f.read()
            txt = [t for t in raw_txt.split('\n') if t.strip()]
        #获取标题信息，标题会加入到一个passage的所有window中
        title = re.search('[A-Za-z_]+[A-Za-z]',txt[0]).group().split('-')+txt[1].strip().split()
        title = " ".join(title)
        title = tokenizer.tokenize(title)
        ntxt = ' '.join(txt[3:])
        ntxt1 = tokenizer.tokenize(ntxt)
        ntxt2 = tokenizer.convert_tokens_to_string(ntxt1)
        offset = raw_txt.index(txt[3])
        entities,relations = aligment_ann(raw_txt[offset:],ntxt2,ann_path,offset)
        #下面把entities level变为wordpiece level
        entities = char_to_wordpiece(ntxt2,entities,tokenizer)
        if is_test:
            #如果我们是需要得到测试用的数据，那么只需要passage,entities,relations
            data.append({"title":title,"passage":ntxt1,"entities":entities,"relations":relations})
        else:
            block_er = get_block_er(ntxt1,entities,relations,window_size,overlap)
            for ber in block_er:
                data.append(block2qas(ber,dataset_tag,question_templates,title))
    save_path = os.path.join(output_dir,os.path.split(data_dir)[-1]+".json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def get_mini_data(path,samples=100):
    with open(path) as f:
        data = json.load(f)
    minidata = data[:samples]
    d,f = os.path.split(path)
    f = "{}_mini_{}".format(samples,f)
    p = os.path.join(d,f)
    with open(p,'w') as f:
        json.dump(minidata,f)

def get_one_passage_data(txt_path,ann_path,output_path,tokenizer,window_size,overlap,dataset_tag,question_templates):
    """给定一个txt和ann文件，我们输出对应的train,dev，test文件"""
    train_data = []
    test_data = []
    with open(txt_path,encoding='utf-8') as f:
        raw_txt = f.read()
        txt = [t for t in raw_txt.split('\n') if t.strip()]
    #获取标题信息，标题会加入到一个passage的所有window中
    title = re.search('[A-Za-z_]+[A-Za-z]',txt[0]).group().split('-')+txt[1].strip().split()
    title = " ".join(title)
    title = tokenizer.tokenize(title)
    ntxt = ' '.join(txt[3:])
    ntxt1 = tokenizer.tokenize(ntxt)
    ntxt2 = tokenizer.convert_tokens_to_string(ntxt1)
    offset = raw_txt.index(txt[3])
    entities,relations = aligment_ann(raw_txt[offset:],ntxt2,ann_path,offset)
    #下面把entities level变为wordpiece level
    entities = char_to_wordpiece(ntxt2,entities,tokenizer)
    test_data.append({"title":title,"passage":ntxt1,"entities":entities,"relations":relations})
    block_er = get_block_er(ntxt1,entities,relations,window_size,overlap)
    for ber in block_er:
        train_data.append(block2qas(ber,dataset_tag,question_templates,title))
    file_name =  os.path.split(txt_path)[-1].split('.')[0]
    json.dump(train_data,open(os.path.join(output_path,file_name+'_train.json'),'w'))
    json.dump(test_data,open(os.path.join(output_path,file_name+'_test.json'),'w'))

def get_one_synthetic_data(tokenizer,output_dir,question_templates):
    train_data = []
    test_data = []
    title = ["af","##p","_","eng","news","story"]
    ntxt1 = tokenizer.tokenize("this is a fake story, so it is very short.")
    ent1_str = tokenizer.convert_tokens_to_string(ntxt1[1:2])
    ent2_str = tokenizer.convert_tokens_to_string(ntxt1[4:7])
    entities = [[ "LOC",1, 2, ent1_str],["GPE",4,7,ent2_str]]
    relations = [["PER-SOC", 0, 1 ]]
    entities1 = [("LOC",1, 2, ent1_str),("GPE",4,7,ent2_str)]
    relations1 = [("PER-SOC", entities1[0], entities1[1])]
    test_data.append({"title":title,"passage":ntxt1,"entities":entities,"relations":relations})
    train_data.append(block2qas([ntxt1,entities1,relations1],'ace2005',question_templates,title))
    json.dump(train_data,open(os.path.join(output_dir,'one_fake_train.json'),'w'))
    json.dump(test_data,open(os.path.join(output_dir,'one_fake_test.json'),'w'))

if __name__=="__main__":
    #data_dir = "./data/raw_data/ACE2005/test"
    txt_path = './data/raw_data/ACE2005/train/AFP_ENG_20030305.0918.txt'
    ann_path = './data/raw_data/ACE2005/train/AFP_ENG_20030305.0918.ann'
    output_dir = "./data/cleaned_data/ACE2005/bert_base_uncased"
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    question_templates = ace2005_question_templates
    #process(data_dir, output_dir, tokenizer, True, 150, 50, 'ace2005',question_templates)
    #get_mini_data("./data/cleaned_data/ACE2005/bert_base_uncased/train.json",1)
    #get_one_passage_data(txt_path,ann_path,output_dir,tokenizer,150,50,'ace2005',question_templates)
    get_one_synthetic_data(tokenizer,output_dir,question_templates)