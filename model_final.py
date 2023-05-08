import os
import torch
import torch.nn as nn
import pickle

from utils.misc import idx_to_one_hot,batch_idx_to_one_hot,drop_padding_triple
import time
from transformers import  ViltModel
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

from utils.BiGRU import GRU, BiGRU

PAD_TRIP=torch.tensor([-1])

PRINT_DETAIL=False



class VisionReasoningNet(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        num_ent = vocab['num_ent']
        num_rel = vocab['num_rel']
        self.node2ans = torch.tensor(vocab['node2answer'])
        self.node2ans_rev = torch.tensor(vocab['node2answer_rev'])
        # num_words = len(vocab['word2id'])
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.num_steps = args.num_steps
        self.min_score = args.min_score

        self.lossqaratio = 100
        print('self.lossqaratio')
        print(self.lossqaratio)

        self.ViltModel=ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        #self.ViltModel = ViltModel.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        dim_word=self.ViltModel.config.hidden_size
        dim_emb = dim_word





        ##Embedding layer as encoder

        self.ent_embedding = nn.Embedding(num_ent, dim_emb)
        self.rel_embedding = nn.Embedding(num_rel, dim_emb)


        #ablation
        # print('Random Initialize ent/rel embedding')

        if args.pretrained_embedding:
            print('Initialize ent/rel embedding from pretrained BERT')
            with open(os.path.join(args.input_dir, 'pretrained_emb_ent.pt'), 'rb') as f:
                self.ent_embedding.weight.data = torch.Tensor(pickle.load(f))
            with open(os.path.join(args.input_dir, 'pretrained_emb_rel.pt'), 'rb') as f:
                self.rel_embedding.weight.data = torch.Tensor(pickle.load(f))





        self.rel_classifier = nn.Linear(dim_emb, 1)
        self.ent_classifier = nn.Linear(dim_emb, 1)
        self.topic_classifier = nn.Linear(dim_emb, 1)
        self.type_classifier = nn.Linear(dim_emb, 1)

        # self.hop_selector = nn.Linear(dim_emb, self.num_steps * 2)
        self.hop_selector = nn.Linear(dim_emb, self.num_steps)

        self.anstype_selector = nn.Linear(dim_emb, 2)




        self.topic_encoder = nn.Sequential(nn.Linear(dim_word, dim_emb), nn.Tanh())
        self.step_encoders = []
        for i in range(self.num_steps):
            m = nn.Sequential(
                nn.Linear(dim_word, dim_emb),
                nn.Tanh(),
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m)

    def follow(self,e, pair, t_prob):
        """
        Args:
            e [num_ent]: previous entity scores
            pair [rsz, 3]: pairs that are taken into consider
            t_prob [esz]: transfer probabilities to neighbour entities
        """
        sub, obj = pair[:, 0], pair[:, 2]
        obj_p = e[sub] * t_prob
        out = torch.index_add(torch.zeros_like(e), 0, obj, obj_p)
        return out

    def forward(self,questions, processed_image, scenegraphs, kb, keyconcepts, answers=None, hops=None,anstype=None):
        #questions [bsz,len_seq]
        #processed_image [bsz,C,H,W]
        #PAD=-1
        #scenegraphs [bsz,max_sg_len,3]
        #kb [bsz,max_kg_len,3]
        #keyconcepts [bsz,max_key_len]
        #answers [bsz,num_ent+num_rel]
        #hops [bsz,num_steps*2]

        #qtype [bsz,2]


        #get question&image embedding via ViLT
        question_lens=questions.size(1)
        _, C, H, W = processed_image.shape

        vilt_input={'input_ids':questions,'pixel_values':processed_image}
        outputs=self.ViltModel(**vilt_input)


        text_emb=outputs.pooler_output
        q_word=outputs.last_hidden_state[:,1:question_lens]




        device=text_emb.device
        bsz, dim_words = text_emb.size()
        num_ent=self.num_ent




  
        



        padtoken = -1
        laste = batch_idx_to_one_hot(keyconcepts, num_ent, padtoken, device)
        z = laste.sum(1).detach().unsqueeze(1)
        margin = (1/z)-1e-6


        

        laste = F.normalize(laste, p=1, dim=1)
        topic_probs = []
        topic_probs.append(laste.clone())

        # margin=self.min_score


        reasoning_graph=[]


        for i in range(bsz):
            ag=torch.cat((drop_padding_triple(scenegraphs[i]), drop_padding_triple(kb[i])), 0)

            ag = ag[ag[:, 0].sort(0).indices]
            reasoning_graph.append(ag)




        ent_probs = []
        rel_probs = []
        for t in range(self.num_steps):

            cq_t = self.step_encoders[t](text_emb)  # [bsz,dim_word]=>[bsz, dim_h]
            q_logits = torch.sum(cq_t.unsqueeze(1) * q_word, dim=2)  # [bsz, dim_h] element* [bsz, max_q, dim_word](need dim_h=dim_word)=>[bsz,max_q]
            q_dist = torch.softmax(q_logits, 1).unsqueeze(1)  # [bsz, 1, max_q]
            #q_dist = q_dist * questions.ne(0).float().unsqueeze(1)
            q_dist = q_dist / (torch.sum(q_dist, dim=2, keepdim=True) + 1e-6)  # [bsz, 1, max_q]
            ctx = (q_dist @ q_word).squeeze(1)  # [bsz, 1, max_q]*[bsz,max_q,dim_word]   =>[bsz, dim_h]
            ctx = ctx + cq_t

            e_stack = []
            r_stack = []


            for i in range(bsz):


                sort_score, sort_idx = torch.sort(laste[i], dim=0, descending=True)
                activated_entidx = sort_idx[sort_score.gt(margin[i])]#.tolist()
                # activated_entidx = sort_idx[sort_score.gt(margin)]  # .tolist()

                target_idx=[]
                for trip in range(len(reasoning_graph[i])):

                    if reasoning_graph[i][trip,0] in activated_entidx :
                        target_idx.append(trip)
                activated_graph=reasoning_graph[i][target_idx]



                rel_id = activated_graph[:, 1]
                ent_id = activated_graph[:, 2]

                ent_emb = self.ent_embedding(ent_id)
                rel_emb = self.rel_embedding(rel_id)


                r_prob = torch.sigmoid(self.rel_classifier(ctx[i:i + 1] * rel_emb).squeeze(1))
                e_prob = torch.sigmoid(self.ent_classifier(ctx[i:i + 1] * ent_emb).squeeze(1))
                type_score = torch.sigmoid(self.type_classifier(ctx[i:i + 1]))
                t_prob = (type_score.squeeze(1) * r_prob + (1 - type_score.squeeze(1)) * e_prob)
                e_stack.append(self.follow(laste[i], activated_graph, t_prob))

                rprob_pad = torch.index_add(torch.zeros(self.num_rel).to(device), 0, rel_id, t_prob)

                r_stack.append(rprob_pad)


            lastr = torch.stack(r_stack, dim=0)
            laste = torch.stack(e_stack, dim=0)



            lastr=F.normalize(lastr, p=1, dim=1)
            laste=F.normalize(laste, p=1, dim=1)

            rel_probs.append(lastr)
            ent_probs.append(laste)

        ent_prob = torch.stack(ent_probs, dim=1)  # [bsz, num_hop, num_ent]
        rel_prob = torch.stack(rel_probs, dim=1)  # [bsz, num_hop, num_rel]

        # hop_logit = self.hop_selector(text_emb)
        # hop_attn = torch.softmax(hop_logit, dim=1)  # [bsz, num_hop*2]
        # final_prob = torch.cat((torch.cat((ent_prob, torch.zeros(bsz, self.num_steps, self.num_rel).to(device)), dim=2),
        #                         torch.cat((torch.zeros(bsz, self.num_steps, self.num_ent).to(device), rel_prob),
        #                                   dim=2)), dim=1)
        # # print(final_prob.shape) [bsz,num_hop*2,num_ent+num_rel] ent+rel
        # pred = torch.sum(final_prob * hop_attn.unsqueeze(2), dim=1)

        hop_logit = self.hop_selector(text_emb)
        hop_attn = torch.softmax(hop_logit, dim=1)  # [bsz, num_hop]

        anstype_logit = self.anstype_selector(text_emb)
        anstype_attn = torch.softmax(anstype_logit, dim=1)  # [bsz, 2]


        final_entprob=torch.sum(ent_prob*hop_attn.unsqueeze(2),dim=1) #[bsz,num_ent]
        final_relprob=torch.sum(rel_prob*hop_attn.unsqueeze(2),dim=1)
        final_prob=torch.cat((torch.cat((final_entprob, torch.zeros(bsz, self.num_rel).to(device)), dim=1).unsqueeze(1),
                                torch.cat((torch.zeros(bsz, self.num_ent).to(device), final_relprob), dim=1).unsqueeze(1)), dim=1)

        pred=torch.sum(final_prob * anstype_attn.unsqueeze(2), dim=1)


        # for i in range(bsz):
        #     if hop_logit[0]>hop_logit[1]:
        #         final_entprob=ent_prob[0]




        #0103 hop&anstype
        if not self.training:
            return {'qa_pred':pred,'hop_pred':hop_logit,'anstype_pred':anstype_logit,'topic_prob':topic_probs[0],
                    'ent_prob_seq':ent_prob,'rel_prob_seq':rel_prob,'margin':margin}
        else:


            # qa loss(v1)
            # weight = answers * 99 + 1
            # loss_score = torch.mean(weight * torch.pow(pred - answers, 2))
            # loss = {'loss_score': 100 * loss_score}


            weight = answers * 99 + 1
            loss_score = torch.mean(weight * torch.pow(pred - answers, 2))
            loss = {'loss_score':  self.lossqaratio * loss_score }

            #hop select loss

            if hops!=None:
                loss_hop=nn.CrossEntropyLoss()(hop_logit, hops)
                loss['loss_hop']=0.01 * loss_hop

            # anstype select loss


            if anstype != None:
                loss_anstype = nn.CrossEntropyLoss()(anstype_logit, anstype)
                loss['loss_anstype'] = 0.01 * loss_anstype
                # loss['loss_anstype'] = 0.0 * loss_anstype



            return loss
