import json

window_size = 100
overlap = 50
query_template_path = './data/query_templates/ace2005.json'
pretrained_model_path = '/home/wangnan/pretrained_models/bert-base-uncased'

tag_idxs = {'B': 0, 'M': 1, 'E': 2, 'S': 3, 'O': 4}


ace2004_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2004_relations = ['ART', 'EMP-ORG', 'GPE-AFF', 'OTHER-AFF', 'PER-SOC', 'PHYS']

ace2005_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2005_entities_full = ["facility","geo political","location","organization","person","vehicle","weapon"]
ace2005_relations = ['ART', 'GEN-AFF', 'ORG-AFF', 'PART-WHOLE', 'PER-SOC', 'PHYS']
ace2005_relations_full = ["artifact","gen affilliation",'organization affiliation','part whole','person social','physical']


ace2005_idx1 = {'FAC': 0, 'GPE': 1, 'LOC': 2, 'ORG': 3, 'PER': 4, 'VEH': 5, 'WEA': 6}
ace2005_idx2t = {}
for i,rel in enumerate(ace2005_relations):
    for j,ent in enumerate(ace2005_entities):
        ace2005_idx2t[(rel,ent)]=i*len(ace2005_relations)+j
ace2005_idx2 = ace2005_idx2t
#这个是训练集上统计得到的分布
ace2005_dist = [[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   3.,   1.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,  33., 116.,  39.,   2.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  10.,  22.,  11.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
       [ 30.,   0.,   0.,   0.,   0.,  60.,  61.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,  19.,   0.,   0.,   1., 143.,  47.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   4.,  14.,   9.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 120.,  31.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   3.,  18.,   8.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
       [ 35.,   0.,   0.,   0.,   0.,  35.,  10., 149.,  20.,   0.,   0.,
          0.,   0.,   5.,   0.,  12.,   0.,   0.,   0., 147.,   1.,  81.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   3.,   5.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
       [ 67.,   1.,   0.,   2.,   0., 113.,  77., 270.,  27.,  10.,  32.,
          0.,   0., 587.,   0., 844.,   5.,   0.,   0.,   0.,   0.,   4.,
          0.,   0.,   0.,   0.,   0.,   4., 434.,   0., 281., 494., 213.,
          4.,   0.,   1.,   0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   8.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   2.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]]

ace2005_question_templates = json.load(open('/home/wangnan/mtqa4kg/data/query_templates/ace2005.json'))