import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict

from utils.misc import idx_to_one_hot, add_kbreverse_rel_idx_to_one_hot, kbidx2reverse
from data_final import load_image

from transformers import ViltProcessor
from transformers import CLIPProcessor

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")




def distributed_concat(tensor):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat


def validate(args, model, data, device):
    vocab = model.module.vocab
    model.eval()
    count = defaultdict(int)
    correct = defaultdict(int)
    print('in validate')
    print(len(data))
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            questions, graphs, answers, hops, anstypes, scenegraphs, keyconcepts, kb, qtype = batch


            batchimage = load_image(pic_dir=args.pic_dir, resize_shape=[512, 384], id2dir=vocab['imageid2url'],
                                    imageid=graphs)
            processed_image = processor(batchimage, [""] * len(batchimage), return_tensors="pt")['pixel_values'].to(
                device)


            questions = questions.to(device)  # tensor [bsz,len_text_seq] on gpu
            mans = (anstypes * vocab['num_ent'] + answers).unsqueeze(1)

            # anstype: ent-0 rel-1
            answers = idx_to_one_hot(mans, vocab['num_ent'] + vocab['num_rel']).to(device)  # [bsz,num_ent+num_rel]
            answers[:, vocab['num_ent'] + 3357:] = answers[:, vocab['num_ent']:vocab['num_ent'] + 3357]

            # 0103 hop&anstype
            # hops = idx_to_one_hot((hops + args.num_steps * anstypes - 1).unsqueeze(1), args.num_steps * 2).to(
            #     device)  # [bsz,num_step*2]
            hops = idx_to_one_hot((hops - 1).unsqueeze(1), args.num_steps).to(device)  # [bsz,num_step]
            anstypes = idx_to_one_hot(anstypes.unsqueeze(1), 2).to(device)  # [bsz,2]

            scenegraphs = scenegraphs.to(device)
            keyconcepts = keyconcepts.to(device)
            kb = kb.to(device)




            outputs = model(questions, processed_image, scenegraphs, kb, keyconcepts)
            """
            outputs: {
            'qa_pred':pred, [bsz,ans]
            'hop_pred':hop_logit, [bsz,num_steps*2]
            }
            """

            if 'qa_pred' in outputs.keys():
                e_score = outputs['qa_pred']
                scores, idx = torch.max(e_score, dim=1)  # [bsz], [bsz]
                match_score = torch.gather(answers, 1, idx.unsqueeze(-1)).squeeze().tolist()
                count['qa'] += len(match_score)
                correct['qa'] += sum(match_score)

            # hop
            # 0103 hop&anstype
            if 'hop_pred' in outputs.keys():
                e_score = outputs['hop_pred']
                scores, idx = torch.max(e_score, dim=1)  # [bsz], [bsz]
                match_score = torch.gather(hops, 1, idx.unsqueeze(-1)).squeeze().tolist()
                count['hop'] += len(match_score)
                correct['hop'] += sum(match_score)

            if 'anstype_pred' in outputs.keys():
                e_score = outputs['anstype_pred']
                scores, idx = torch.max(e_score, dim=1)  # [bsz], [bsz]
                match_score = torch.gather(anstypes, 1, idx.unsqueeze(-1)).squeeze().tolist()
                count['anstype'] += len(match_score)
                correct['anstype'] += sum(match_score)


    return correct, count


def predict(args, model, data, device):
    vocab = model.vocab
    model.eval()
    count = defaultdict(int)
    correct = defaultdict(int)
    # count=0
    # correct=0
    # print('in predict')
    # print(len(data))
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            questions, graphs, answers, hops, anstypes, scenegraphs, keyconcepts, kb, qtype, topic, ent_seq, rel_seq = batch


            batchimage = load_image(pic_dir=args.pic_dir, resize_shape=[512, 384], id2dir=vocab['imageid2url'],
                                    imageid=graphs)
            processed_image = processor(batchimage, [""] * len(batchimage), return_tensors="pt")['pixel_values'].to(
                device)

            #for ablation
            # bsz = questions.size(0)
            # processed_image = torch.zeros(bsz, 3, 384, 512).to(device)

            questions = questions.to(device)  # tensor [bsz,len_text_seq] on gpu
            mans = (anstypes * vocab['num_ent'] + answers).unsqueeze(1)

            # anstype: ent-0 rel-1
            answers = idx_to_one_hot(mans, vocab['num_ent'] + vocab['num_rel']).to(device)  # [bsz,num_ent+num_rel]
            answers[:, vocab['num_ent'] + 3357:] = answers[:, vocab['num_ent']:vocab['num_ent'] + 3357]



            # hops = idx_to_one_hot((hops + args.num_steps * anstypes - 1).unsqueeze(1), args.num_steps * 2).to(
            #     device)  # [bsz,num_step*2]
            hops = idx_to_one_hot((hops - 1).unsqueeze(1), args.num_steps).to(device)  # [bsz,num_step]
            anstypes = idx_to_one_hot(anstypes.unsqueeze(1), 2).to(device)  # [bsz,2]

            scenegraphs = scenegraphs.to(device)
            keyconcepts = keyconcepts.to(device)
            kb = kb.to(device)
            qtype = qtype.to(device)



            topic = idx_to_one_hot(topic.unsqueeze(1), vocab['num_ent']).to(device)
            ent_1 = idx_to_one_hot(ent_seq[:, :1], vocab['num_ent']).to(device)
            ent_2 = idx_to_one_hot(ent_seq[:, 1:], vocab['num_ent']).to(device)
            rel_1 = idx_to_one_hot(rel_seq[:, :1], vocab['num_rel']).to(device)
            rel_2 = idx_to_one_hot(rel_seq[:, 1:], vocab['num_rel']).to(device)




            outputs = model(questions, processed_image, scenegraphs, kb, keyconcepts)
            """


            outputs: {
            'qa_pred':pred, [bsz,ans]
            'hop_pred':hop_logit, [bsz,num_steps*2]
            
            'topic_prob':topic_prob [bsz,num_ent]
            'ent_prob_seq':ent_prob [bsz,num_steps,num_ent]
            'rel_prob_seq':rel_prob [bsz,num_steps,num_rel]
            }
            """

           


            if 'hop_pred' in outputs.keys():
                e_score = outputs['hop_pred']
                scores, idx = torch.max(e_score, dim=1)  # [bsz], [bsz]
                match_score = torch.gather(hops, 1, idx.unsqueeze(-1)).squeeze().tolist()
                count['hop'] += len(match_score)
                correct['hop'] += sum(match_score)

            if 'anstype_pred' in outputs.keys():
                e_score = outputs['anstype_pred']
                scores, idx = torch.max(e_score, dim=1)  # [bsz], [bsz]
                match_score = torch.gather(anstypes, 1, idx.unsqueeze(-1)).squeeze().tolist()
                count['anstype'] += len(match_score)
                correct['anstype'] += sum(match_score)


            if 'qa_pred' in outputs.keys():
                e_score = outputs['qa_pred']
                scores, idx = torch.max(e_score, dim=1)  # [bsz], [bsz]
                match_score = torch.gather(answers, 1, idx.unsqueeze(-1)).squeeze().tolist()
                # print(len(match_score))
                count['qa_all'] += len(match_score)
                correct['qa_all'] += sum(match_score)

                # print(qtype)
                # print('match_score')
                # print(match_score)


                for point in range(len(match_score)):
                    question_type = [qtype[point][0].item(), qtype[point][1].item()]
                    count['qa_{}-{}'.format(question_type[0], question_type[1])] += 1
                    correct['qa_{}-{}'.format(question_type[0], question_type[1])] += match_score[point]



            e_score = outputs['topic_prob']
            scores, idx = torch.max(e_score, dim=1)  # [bsz], [bsz]
            match_score = torch.gather(topic, 1, idx.unsqueeze(-1)).squeeze().tolist()
            # print(len(match_score))
            count['topic'] += len(match_score)
            correct['topic'] += sum(match_score)

            e_score = outputs['ent_prob_seq'][:,0,:]
            scores, idx = torch.max(e_score, dim=1)  # [bsz], [bsz]
            match_score = torch.gather(ent_1, 1, idx.unsqueeze(-1)).squeeze().tolist()
            # print(len(match_score))
            count['ent_1'] += len(match_score)
            correct['ent_1'] += sum(match_score)

            e_score = outputs['ent_prob_seq'][:, 1, :]
            scores, idx = torch.max(e_score, dim=1)  # [bsz], [bsz]
            match_score = torch.gather(ent_2, 1, idx.unsqueeze(-1)).squeeze().tolist()
            # print(len(match_score))
            count['ent_2'] += len(match_score)
            correct['ent_2'] += sum(match_score)

            e_score = outputs['rel_prob_seq'][:, 0, :]
            scores, idx = torch.max(e_score, dim=1)  # [bsz], [bsz]
            match_score = torch.gather(rel_1, 1, idx.unsqueeze(-1)).squeeze().tolist()
            # print(len(match_score))
            count['rel_1'] += len(match_score)
            correct['rel_1'] += sum(match_score)

            e_score = outputs['rel_prob_seq'][:, 1, :]
            scores, idx = torch.max(e_score, dim=1)  # [bsz], [bsz]
            match_score = torch.gather(rel_2, 1, idx.unsqueeze(-1)).squeeze().tolist()
            # print(len(match_score))
            count['rel_2'] += len(match_score)
            correct['rel_2'] += sum(match_score)


            correct['step_1'] += ( correct['ent_1'] + correct['rel_1'])
            correct['step_2'] += ( correct['ent_2'] + correct['rel_2'])
            count['step_1']+=count['rel_1']
            count['step_2']+=count['rel_2']

    return correct, count
