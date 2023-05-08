import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
import numpy as np
from tqdm import tqdm
import time
from utils.misc import MetricLogger, load_glove, idx_to_one_hot, RAdam,SmoothedValue,add_kbreverse_rel_idx_to_one_hot,kbidx2reverse
import json
from data_final import DataLoader4test,load_vocab,load_image,DataLoader

#0115 ablation
from model_final import VisionReasoningNet
from predict_final import validate
import logging
from transformers import  ViltProcessor
from transformers import  CLIPProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

processor=ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")


torch.set_num_threads(1)  #
#For DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def train(args):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert torch.cuda.is_available()
    #DDP
    device=args.local_rank
    print(device)
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

    logging.info("Create train_loader and val_loader ........")

    vocab_json = os.path.join(args.input_dir, 'vocab_final.json')


    #1226_topic_candidate
    train_pt = os.path.join(args.input_dir, 'train_topic_candidate.pt')
    val_pt = os.path.join(args.input_dir, 'val_topic_candidate.pt')


    train_loader = DataLoader4test(train_pt, args.batch_size,  distribute=True)

    val_loader = DataLoader4test(val_pt, args.batch_size//8,distribute=True)
    vocab = load_vocab(vocab_json)


    logging.info("Create model.........")


    #load pretrained node embedding & entity embedding
    model = VisionReasoningNet(args, vocab)


    if dist.get_rank() == 0 and args.ckpt is not None:
        logging.info("Load ckpt from {}".format(args.ckpt))
        model.load_state_dict(torch.load(args.ckpt))


    model = model.to(device)

    # DDP
    logging.info(model.args)
    model = DDP(model,find_unused_parameters=True ,device_ids=[device], output_device=device)


    logging.info(model)

    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'radam':
        optimizer = RAdam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=1)

    meters = MetricLogger(delimiter="  ")
    # logging.info("Test validate........")
    # correct, count = validate(args, model, val_loader, device)

    logging.info("Start training........")

    best_qa=0.0
    for epoch in range(args.num_epoch):
        model.train()
        train_loader.sampler.set_epoch(epoch)

        for iteration, batch in enumerate(train_loader):
            iteration = iteration + 1
            questions, graphs, answers, hops, anstypes, scenegraphs, keyconcepts, kb, qtype = batch


            # batchimage = load_image(pic_dir=args.pic_dir, resize_shape=[512, 384], id2dir=vocab['imageid2url'], imageid=graphs)
            # processed_image=processor(batchimage, [""]*len(batchimage), return_tensors="pt")['pixel_values'].to(device)

            bsz = questions.size(0)
            processed_image = torch.zeros(bsz, 3, 384, 512).to(device)

            # print(processed_image.shape)
            # bsz=questions.size(0)
            # processed_image=torch.zeros(bsz, 3, 384, 512).to(device)
            # input()
            # processed_image=0

            questions = questions.to(device)#tensor [bsz,len_text_seq] on gpu
            mans=(anstypes*vocab['num_ent']+answers).unsqueeze(1)

            # anstype: ent-0 rel-1
            answers = idx_to_one_hot(mans, vocab['num_ent']+vocab['num_rel']).to(device)#[bsz,num_ent+num_rel]
            answers[:,vocab['num_ent']+3357:]=answers[:,vocab['num_ent']:vocab['num_ent']+3357]


            #0103 hop&anstype
            # hops=(hops+args.num_steps*anstypes-1).to(device)
            hops=(hops-1).to(device)
            anstypes=anstypes.to(device)

   

            scenegraphs = scenegraphs.to(device)
            keyconcepts = keyconcepts.to(device)
            kb = kb.to(device)



            loss = model(questions, processed_image, scenegraphs, kb, keyconcepts, answers, hops, anstypes)

            optimizer.zero_grad()
            if isinstance(loss, dict):
                total_loss = sum(loss.values())
                meters.update(**{k: v.item() for k, v in loss.items()})
            else:
                total_loss = loss
                meters.update(loss=loss.item())
            total_loss.backward()


            nn.utils.clip_grad_value_(model.parameters(), 0.5)
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()

            if iteration % (len(train_loader) // 50) == 0:
                logging.info(
                    meters.delimiter.join(
                        [
                            "progress: {progress:.3f}",
                            "{meters}",
                            "lr: {lr:.6f}",
                        ]
                    ).format(
                        progress=epoch + iteration / len(train_loader),
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )
        if (epoch + 1) % int(1 / args.valid_ratio) == 0 :
            #validate on train for convergence testing
            # correct, count = validate(args, model, train_loader, device)
            # tensor = torch.tensor([[count['qa']], [correct['qa']], [correct['hop']], [correct['anstype']]]).to(device)
            # output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
            #
            # torch.distributed.all_gather(output_tensors, tensor)
            # # print(output_tensors)
            # concat = torch.cat(output_tensors, dim=1).sum(1)
            # # print(concat)
            # acc = {'qa': (concat[1] / concat[0]).item(), 'hop': (concat[2] / concat[0]).item(),
            #        'anstype': (concat[3] / concat[0]).item()}
            # # print(acc)
            # logging.info('Train set ACC')
            # logging.info(acc)



            #validate on val set

            correct,count = validate(args, model, val_loader, device)
            tensor = torch.tensor([ [count['qa']], [correct['qa']], [correct['hop']], [correct['anstype']] ]).to(device)
            output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]

            torch.distributed.all_gather(output_tensors, tensor)
            #print(output_tensors)
            concat = torch.cat(output_tensors, dim=1).sum(1)
            #print(concat)

            acc={'qa':(concat[1]/concat[0]).item(),'hop':(concat[2]/concat[0]).item(),'anstype':(concat[3]/concat[0]).item()}
            #print(acc)
            logging.info('Val set ACC')
            logging.info(acc)
            scheduler.step(acc['qa'])


            if dist.get_rank()==0:
                if acc['qa']>=best_qa:
                    best_qa=acc['qa']
                    torch.save(model.module.state_dict(),
                               os.path.join(args.log_model_path, 'model_epoch-{}_acc-{:.4f}.pt'.format(epoch+1, acc['qa'])))
                    logging.info('model saved')


def main():
    parser = argparse.ArgumentParser()
    # input and output
    #DDP
    parser.add_argument("--local_rank", default=-1,type=int)

    parser.add_argument('--input_dir', default='./keydata/KRVQA-christmas')
    parser.add_argument('--pic_dir', default='./keydata/KRVQA')
    parser.add_argument('--save_dir', default='./save_dir-christmas', help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default=None)
    # training parameters
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=30, type=int)
    #bsz<=16
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--opt', default='radam', type=str)
    parser.add_argument('--valid_ratio', default=0.5, type=float)
    # model hyperparameters

    parser.add_argument('--num_steps', default=2, type=int)

    #原本是定值。平均初始化后改为初始值的一个比例
    parser.add_argument('--min_score', default=0.7, type=float,
                        help='activate an entity when its score exceeds this proportion of value')  # 0.9 may cause convergency issue


    parser.add_argument('--pretrained_embedding',  default=1, choices=[0, 1])
    args = parser.parse_args()

    # make logging.info display into both shell and file
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    args.log_name = time_ + '_{}_{}_{}.log'.format(args.opt, args.lr, args.batch_size)
    args.log_model_path = os.path.join(args.save_dir, time_)
    if not os.path.exists(args.log_model_path):
        os.makedirs(args.log_model_path)
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, args.log_name))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k + ':' + str(v))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)




if __name__ == '__main__':
    main()
