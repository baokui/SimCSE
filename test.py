from simcse import SimCSE
import json
import numpy as np
path_model = "/search/odin/guobk/data/simcse/simcse_roberta_zh_l12"
model = SimCSE(path_model)
with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs.json','r') as f:
    Docs = json.load(f)
with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries-142000.json','r') as f:
    Queries = json.load(f)
S = [Docs[i]['content'] for i in range(len(Docs))]
V = model.encode(S)
for i in range(len(Docs)):
    # c = Docs[i]['content']
    v = V[i]
    v = v.numpy()
    #v = np.float64(v)
    d = v.dot(v)
    v = list(v)
    Docs[i]['sent2vec'] = v
    if i%100==0:
        print(i,len(Docs),d)
V = [Docs[i]['sent2vec'] for i in range(len(Docs))]
np.save('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs-simcse.npy',V)
S = [Queries[i]['content'] for i in range(len(Queries))]
V = model.encode(S)
for i in range(len(Queries)):
    # c = Docs[i]['content']
    v = V[i]
    v = v.numpy()
    v = np.float64(v)
    d = v.dot(v)
    v = list(v)
    Queries[i]['sent2vec'] = v
    if i%100==0:
        print(i,len(Queries),d)

# with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries-simcse.json','w') as f:
#     json.dump(Queries,f,ensure_ascii=False,indent=4)
V = [Queries[i]['sent2vec'] for i in range(len(Queries))]
np.save('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries-simcse.npy',V)