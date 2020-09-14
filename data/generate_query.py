import json

ace2004_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2004_relations = ['ART', 'EMP-ORG', 'GPE-AFF', 'OTHER-AFF', 'PER-SOC', 'PHYS']

ace2005_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2005_entities_full = ["facility","geo political","location","organization","person","vehicle","weapon"]
ace2005_relations = ['ART', 'GEN-AFF', 'ORG-AFF', 'PART-WHOLE', 'PER-SOC', 'PHYS']
ace2005_relations_full = ["artifact","gen affilliation",'organization affiliation','part whole','person social','physical']

templates = {"qa_turn1":{},"qa_turn2":{}}
for ent1,ent1f in zip(ace2005_entities,ace2005_entities_full):
    templates['qa_turn1'][ent1]="find all {} entities mentioned in the text.".format(ent1f)
    for rel,relf in zip(ace2005_relations,ace2005_relations_full):
        for ent2,ent2f in zip(ace2005_entities,ace2005_entities_full):
            templates['qa_turn2'][str((ent1,rel,ent2))]="find all {} entities mentioned in the text that have a {} relation with {} entity XXX.".format(ent2f,relf,ent1f)

with open(r"C:\Users\DELL\Desktop\mtqa4kg\data\query_templates\ace2005_query_templates.json",'w') as f:
    json.dump(templates,f)