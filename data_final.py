import json
import pickle
import torch
import numpy as np
import os
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def load_vocab(path):
    vocab=json.load(open(path))
    return vocab

def load_image(pic_dir,resize_shape,id2dir,imageid):
    Batch_Image=[]
    for i in imageid:
        path = id2dir[str(i)]
        width = resize_shape[0]
        height = resize_shape[1]
        Batch_Image.append(Image.open(os.path.join(pic_dir, path)).convert('RGB').resize((width, height),Image.ANTIALIAS))
    return Batch_Image


def collate(batch):
    #questions,graphs,answers,hops,anstypes,scenegraphs,keyconcepts,kb,qtype,topic, ent_seq, rel_seq
    batch = list(zip(*batch))
    question = list(map(torch.stack, batch[:1]))[0]
    graph = list(batch[1])
    answer = torch.LongTensor(batch[2])
    hop = torch.LongTensor(batch[3])
    anstype = torch.LongTensor(batch[4])
    sg = torch.LongTensor(batch[5])
    keyconcept=torch.LongTensor(batch[6])
    kb=torch.LongTensor(batch[7])
    qtype = torch.LongTensor(np.array(batch[8]))
    topic = torch.LongTensor(np.array(batch[9]))
    ent_seq = torch.LongTensor(np.array(batch[10]))
    rel_seq = torch.LongTensor(np.array(batch[11]))
    return question,graph, answer,hop,anstype, sg, keyconcept,kb,qtype,topic,ent_seq,rel_seq

def collate4test(batch):
    #questions,graphs,answers,hops,anstypes,scenegraphs,keyconcepts,kb,qtype
    batch = list(zip(*batch))
    question = list(map(torch.stack, batch[:1]))[0]
    graph = list(batch[1])
    answer = torch.LongTensor(batch[2])
    hop = torch.LongTensor(batch[3])
    anstype = torch.LongTensor(batch[4])
    sg = torch.LongTensor(batch[5])
    keyconcept=torch.LongTensor(batch[6])
    kb=torch.LongTensor(batch[7])
    qtype = torch.LongTensor(np.array(batch[8]))
    return question,graph, answer,hop,anstype, sg, keyconcept,kb,qtype


class Dataset(torch.utils.data.Dataset):
    ## questions,graphs,answers,hops,anstypes,scenegraphs,keyconcepts,kb,qtype,topic, ent_seq, rel_seq
    def __init__(self, inputs):
        self.questions, self.graphs, self.answers, self.hops, self.anstypes,self.scenegraphs,self.keyconcepts,self.kb,self.qtype,self.topic,self.ent_seq,self.rel_seq = inputs

    def __getitem__(self, index):
        question = torch.LongTensor(self.questions[index])
        graph=self.graphs[index]
        anstype = self.anstypes[index]
        answer = self.answers[index]
        hop = self.hops[index]
        sg = self.scenegraphs[index]
        keyconcept=self.keyconcepts[index]
        kb = self.kb[index]
        qtype = self.qtype[index]
        topic = self.topic[index]
        ent_seq = self.ent_seq[index]
        rel_seq = self.rel_seq[index]
        return question, graph, answer, hop,anstype, sg,keyconcept,kb,qtype,topic,ent_seq,rel_seq

    def __len__(self):
        return len(self.questions)

class Dataset4test(torch.utils.data.Dataset):
    #questions,graphs,answers,hops,anstypes,scenegraphs,keyconcepts,kb,qtype,
    def __init__(self, inputs):
        self.questions, self.graphs, self.answers, self.hops, self.anstypes,self.scenegraphs,self.keyconcepts,self.kb ,self.qtype = inputs

    def __getitem__(self, index):
        question = torch.LongTensor(self.questions[index])
        graph=self.graphs[index]
        anstype = self.anstypes[index]
        answer = self.answers[index]
        hop = self.hops[index]
        sg = self.scenegraphs[index]
        keyconcept=self.keyconcepts[index]
        kb=self.kb[index]
        qtype = self.qtype[index]
        return question, graph, answer, hop,anstype, sg,keyconcept,kb,qtype

    def __len__(self):
        return len(self.questions)


# 载入sg前需要padding，让长度对齐。question的padding倒是已经在预处理的时候做好了，但sg的得在dataloader里头做，不然数据缓存太大了
class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, question_pt, batch_size, test_mode=True,distribute=False):
        # questions,graphs,answers,hops,anstypes,scenegraphs,keyconcepts,kb,qtype,topic, ent_seq, rel_seq
        rangesize=12 if(test_mode) else 9

        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(rangesize):
                inputs.append(pickle.load(f))

        max_len = max(len(q) for q in inputs[5])
        print('max SG length: {}'.format(max_len))

        padding = [-1, -1, -1]
        for sg in inputs[5]:

            while len(sg) < max_len:
                sg.append(padding)

        max_len = max(len(q) for q in inputs[6])
        print('max Keyconcept length: {}'.format(max_len))
        padding = -1
        for idx in inputs[6]:
            while len(idx) < max_len:
                idx.append(padding)

        # knowledge graph  padding :token--[-1,-1,-1]
        max_len = max(len(q) for q in inputs[7])
        print('max KGidx length: {}'.format(max_len))

        padding = [-1, -1,-1]
        for idx in inputs[7]:
            while len(idx) < max_len:
                idx.append(padding)




        print('data number: {}'.format(len(inputs[0])))

        dataset = Dataset(inputs)

        #shuffle = training
        if distribute:
            sampler=torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler=None

        super().__init__(
            dataset,
            batch_size=batch_size,
            #shuffle=shuffle,
            num_workers=0,
            collate_fn=collate,
            sampler=sampler
        )

class DataLoader4test(torch.utils.data.DataLoader):
    def __init__(self, question_pt, batch_size, distribute=False,):
        # questions,graphs,answers,hops,anstypes,scenegraphs,keyconcepts,kb,qtype,||||topic, ent_seq, rel_seq
        rangesize = 9

        inputs = []

        with open(question_pt, 'rb') as f:
            for _ in range(rangesize):
                inputs.append(pickle.load(f))



        max_len = max(len(q) for q in inputs[5])
        print('max SG length: {}'.format(max_len))
        padding = [-1, -1, -1]
        for sg in inputs[5]:
            while len(sg) < max_len:
                sg.append(padding)

        max_len = max(len(q) for q in inputs[6])
        print('max Keyconcept length: {}'.format(max_len))
        padding = -1
        for idx in inputs[6]:
            while len(idx) < max_len:
                idx.append(padding)

        # knowledge graph padding :token--[-1,-1,-1]
        max_len = max(len(q) for q in inputs[7])
        print('max KG length: {}'.format(max_len))

        padding = [-1, -1,-1]
        for idx in inputs[7]:
            while len(idx) < max_len:
                idx.append(padding)

        print('data number: {}'.format(len(inputs[0])))

        dataset = Dataset4test(inputs)

        # shuffle = training
        if distribute:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = None

        super().__init__(
            dataset,
            batch_size=batch_size,
            # shuffle=shuffle,
            collate_fn=collate4test,
            sampler=sampler
        )
# self.vocab = vocab

if __name__ == '__main__':
#     vocab_json = '/data/csl/exp/AI_project/SRN/input/vocab.json'
    val_pt = os.path.join('keydata/KRVQA-christmas', 'val_topic_candidate.pt')
    pic_dir = 'keydata/KRVQA'
    # device=-1
    # print(device)
    # torch.cuda.set_device(device)
    # dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

    print("Createal_loader ........")
    val_loader = DataLoader4test(val_pt, 4,distribute=True)

    for iteration, batch in enumerate(val_loader):
        # print(dist.get_rank())
        iteration = iteration + 1

        # questions, anstypes, answers, hops, scenegraphs
        questions, graphs, answers, hops, anstypes, scenegraphs, keyconcepts, kb, qtype = batch
        print(questions)
        print(qtype)
        input()

