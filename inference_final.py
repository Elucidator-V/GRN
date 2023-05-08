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
#ablation
from model_final import VisionReasoningNet
from predict_final import predict
import logging
from transformers import  ViltProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

processor=ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")


torch.set_num_threads(1)  #
#For DDP
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #0110

    print(device)
    # assert torch.cuda.is_available()
    # #DDP
    # device=args.local_rank
    # print(device)
    # torch.cuda.set_device(device)
    # dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

    logging.info("Create test_loader ........")

    vocab_json = os.path.join(args.input_dir, 'vocab_final.json')


    #1226 topic_candidate
    # train_pt = os.path.join(args.input_dir, 'train.pt')
    # val_pt = os.path.join(args.input_dir, 'val.pt')

    # test_pt = os.path.join(args.input_dir, 'test_topic_candidate.pt')
    # ood test

    logging.info("Testing testset")
    test_pt = os.path.join(args.input_dir, 'test_topic_candidate.pt')

    # logging.info("Testing zero-shot subset")
    # test_pt = os.path.join(args.input_dir, 'test_ood.pt')

    # logging.info("Testing 3 cases")
    # test_pt = os.path.join(args.input_dir, 'test_3case.pt')


    # train_loader = DataLoader4test(train_pt, args.batch_size,  distribute=True)
    #
    # val_loader = DataLoader4test(val_pt, args.batch_size//8,distribute=True)
    test_loader = DataLoader(test_pt, args.batch_size, distribute=False)
    vocab = load_vocab(vocab_json)


    logging.info("Create model.........")


    #load pretrained node embedding & entity embedding
    model = VisionReasoningNet(args, vocab)


    if  args.ckpt is not None:
        logging.info("Load ckpt from {}".format(args.ckpt))
        model.load_state_dict(torch.load(args.ckpt))


    model = model.to(device)




    logging.info(model)
    logging.info(model.args)

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5)

    meters = MetricLogger(delimiter="  ")
    # validate(args, model, val_loader, device)
    logging.info("Start testing........")

    correct, count = predict(args, model, test_loader, device)
    acc = {k: correct[k] / count[k] for k in count}
    logging.info(acc)




def main():
    parser = argparse.ArgumentParser()
    # input and output
    #DDP
    parser.add_argument("--local_rank", default=-1,type=int)

    parser.add_argument('--input_dir', default='./keydata/KRVQA-christmas')
    parser.add_argument('--pic_dir', default='./keydata/KRVQA')
    parser.add_argument('--save_dir', default='./save_dir-christmas', help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default=None,required=True)
    # training parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    #bsz<=16
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--opt', default='radam', type=str)
    parser.add_argument('--valid_ratio', default=0.5, type=float)
    # model hyperparameters

    parser.add_argument('--num_steps', default=2, type=int)

    
    parser.add_argument('--min_score', default=0.7, type=float,
                        help='activate an entity when its score exceeds this proportion of value')  # 0.9 may cause convergency issue

    parser.add_argument('--pretrained_embedding',  default=0, choices=[0, 1])
    args = parser.parse_args()

    # make logging.info display into both shell and file
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    args.log_name = time_ + 'Test_{}_{}_{}.log'.format(args.opt, args.lr, args.batch_size)
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
