################################
# supervised simcse - origin
from simcse import SimCSE
import json
import numpy as np
import sys
mode = sys.argv[1]
maxRecall = 20
minSim = 0.0
if mode=='sup':
    modeltag = 'simcse_sup'
    path_model = "/search/odin/guobk/data/simcse/simcse_roberta_zh_l12_sup" 
    path_target="/search/odin/guobk/data/bert_semantic/finetuneData_new_test/result-simcse_sup.json"
else:
    modeltag = 'simcse_upsup'
    path_model = "/search/odin/guobk/data/simcse/simcse_roberta_zh_l12"
    path_target="/search/odin/guobk/data/bert_semantic/finetuneData_new_test/result-simcse_unsup.json"
model = SimCSE(path_model)
with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs.json','r') as f:
    Docs = json.load(f)
with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries.json','r') as f:
    Queries = json.load(f)
Sd = [Docs[i]['content'] for i in range(len(Docs))]
Vd = model.encode(Sd)
Sq = [Queries[i]['content'] for i in range(1000)]
Vq = model.encode(Sq)
Vd = Vd.numpy()
Vq = Vq.numpy()
R = []
for i in range(len(Vq)):
    s = Vd.dot(Vq[i])
    idx = np.argsort(-s)
    rec = [Sd[j]+'\t%0.4f'%s[j] for j in idx[:maxRecall] if s[j]>=minSim]
    d = Queries[i]
    d['rec_'+modeltag] = rec
    R.append(d)
with open(path_target,'w') as f:
    json.dump(R,f,ensure_ascii=False,indent=4)

################################
# unsupervised simcse - origin
# from simcse import SimCSE
# import json
# import numpy as np
# maxRecall = 20
# minSim = 0.0
# modeltag = 'simcse_sup'
# path_model = "/search/odin/guobk/data/simcse/simcse_roberta_zh_l12"
# path_target="/search/odin/guobk/data/bert_semantic/finetuneData_new_test/result-simcse_unsup.json"
# model = SimCSE(path_model)
# with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs.json','r') as f:
#     Docs = json.load(f)
# with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries.json','r') as f:
#     Queries = json.load(f)
# Sd = [Docs[i]['content'] for i in range(len(Docs))]
# Vd = model.encode(Sd)
# Sq = [Queries[i]['content'] for i in range(1000)]
# Vq = model.encode(Sq)
# Vd = Vd.numpy()
# Vq = Vq.numpy()
# R = []
# for i in range(len(Vq)):
#     s = Vd.dot(Vq[i])
#     idx = np.argsort(-s)
#     rec = [Sd[j]+'\t%0.4f'%s[j] for j in idx[:maxRecall] if s[j]>=minSim]
#     d = Queries[i]
#     d['rec_'+modeltag] = rec
#     R.append(d)
# with open(path_target,'w') as f:
#     json.dump(R,f,ensure_ascii=False,indent=4)