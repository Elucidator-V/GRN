import json
import numpy as np
from nltk import word_tokenize
import collections
from collections import Counter, defaultdict
from tqdm import tqdm
import re
import os
import pickle
import torch
# from transformers import ViltProcessor, ViltModel
from transformers import AutoTokenizer


# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
tokenizer=AutoTokenizer.from_pretrained('bert-base-cased')
padtoken=0
# model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

def encode_dataset(vocab, dataset):
    questions = []
    answers = []
    anstypes = []
    hops = []
    scenegraphs = []
    keyconcepts=[]
    graphs=[]
    kb=[]

    qtype = []
    topic = []
    ent_seq = []
    rel_seq = []



    for qa in tqdm(dataset):


        answers.append(qa['answer'])

        #ent-0 rel-1
        if qa['anstype']=='ent':
            anstypes.append(0)
        elif qa['anstype']=='rel':
            anstypes.append(1)
        else:
            print('error in anstype')
        hops.append(qa['hop'])
        graphs.append(qa['image_id'])
        scenegraphs.append(qa['scene_graph'])

        keyconcepts.append(qa['topic_candidate'])
        kb.append(qa['related_kb'])
        topic.append(qa['topic_ent'])
        ent_seq.append(qa['ent_seq'])
        rel_seq.append(qa['rel_seq'])
        qtype.append(qa['question_type'])


        text=qa['question']
        # inputs = processor(image, text, return_tensors="pt")
        textinfo=tokenizer(text)
        questions.append(textinfo['input_ids'])





    # question padding
    max_len = max(len(q) for q in questions)
    print('max question length:{}'.format(max_len))
    # IN ViLT tokenizer(BERT tokenizer) [PAD]->0

    for q in questions:
        while len(q) < max_len:
            q.append(padtoken)





    questions = np.asarray(questions, dtype=np.int32)
    graphs=np.asarray(graphs, dtype=np.int32)
    answers = np.asarray(answers, dtype=np.int32)
    hops = np.asarray(hops, dtype=np.int32)
    anstypes = np.asarray(anstypes, dtype=np.int32)
    scenegraphs = np.asarray(scenegraphs)
    kb = np.asarray(kb)
    keyconcepts = np.asarray(keyconcepts)
    qtype = np.asarray(qtype)
    topic = np.asarray(topic)
    ent_seq = np.asarray(ent_seq)
    rel_seq = np.asarray(rel_seq)

    return questions,graphs,answers,hops,anstypes,scenegraphs,keyconcepts,kb,qtype,topic, ent_seq, rel_seq


with open('keydata/KRVQA-christmas/datasets_3cases.json') as f:
    datasets = f.read()
datasets=json.loads(datasets)
train_set, val_set, test_set = datasets[0], datasets[1], datasets[2]
print('size of training data: {}'.format(len(train_set)))
print('size of test data: {}'.format(len(test_set)))
print('size of valid data: {}'.format(len(val_set)))


with open('keydata/KRVQA-christmas/i2u.json') as f:
    meta = f.read()
id2dir=json.loads(meta)

print('id2dir loaded')

vocab=0


# for name, dataset in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
# for name, dataset in zip(('val_test', 'test'), (val_set, test_set)):
for name, dataset in zip((['test']), ( [test_set])):
    print('Encode {} set'.format(name))

    target=dataset
    outputs = encode_dataset(vocab, target)
    #return questions,graphs,answers,hops,anstypes,scenegraphs,keyconcepts,kgidx
    # with open(os.path.join('keydata/KRVQA-christmas/', '{}_topic_candidate.pt'.format(name)), 'wb') as f:
    with open(os.path.join('keydata/KRVQA-christmas/', '{}_3case.pt'.format(name)), 'wb') as f:
        for o in outputs:
            # print(o.shape)
            pickle.dump(o, f)
            print('saved!')
        outputs=0



