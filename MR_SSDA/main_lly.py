from __future__ import print_function
import argparse
import torch
from solver import Solver
import os
import time
import numpy as np

time_start = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MCD Implementation')
parser.add_argument('--source', type=str, default='230ICIAR-2018', metavar='N',
                    help='source dataset')
parser.add_argument('--target', type=str, default='230BreaKHis200', metavar='N',
                    help='target dataset')
parser.add_argument('--model', type=str, default='BreNet_MLFF', metavar='N',
                    help='name of the model')
parser.add_argument('--expl', type=str, default='', metavar='N',
                    help='explain the adjustment')
parser.add_argument('--MFFA_pt', type=float, default=0.00001, metavar='S',
                    help='weight of MFFA loss')
parser.add_argument('--MRDA_pt', type=float, default=0.000001, metavar='S',
                    help='weight of MRDA loss')
parser.add_argument('--gama', type=float, default=0.9, metavar='S',
                    help='learning rate attenuation coefficient (default: 0.9)')
parser.add_argument('--MLFF_dim', type=int, default=512, metavar='S',
                    help='dimensionality reduction')
parser.add_argument('--lb_rate', type=int, default=5, metavar='S',  # 5 or 13
                    help='rate of labeled dataset')
parser.add_argument('--ts_rate', type=float, default=0.63, metavar='S',
                    help='rate of target and source')
parser.add_argument('--batch_size_tra', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--batch_size_tes', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--num_k', type=int, default=2, metavar='N',
                    help='hyper paremeter for generator update')
parser.add_argument('--max_epoch', type=int, default=30, metavar='N',
                    help='how many epochs')
parser.add_argument('--save_epoch', type=int, default=29, metavar='N',
                    help='when to save the model')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num_workers', type=int, default=4, metavar='S',
                    help='the number of processes loaded with multiple processes')
parser.add_argument('--pin_memory', action='store_false', default=True,
                    help='locked page memory')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args, '\n')


def main():
    # if not args.one_step:
    record_name = '%s_%s_%s_%s_%s' % (args.source.split('/')[-1], args.target.split('/')[-1], args.model, args.tlabel, args.expl)
    if not os.path.exists('record/%s' % record_name):
        os.mkdir('record/%s' % record_name)

    record_num = 0
    list_F1s = []
    record_train = 'record/%s/tsrate_%s_batch_tra_%s_batch_tes_%s_lr_%s_seed_%s_%s.txt' % (
    record_name, args.ts_rate, args.batch_size_tra, args.batch_size_tes, args.lr, args.seed, record_num)
    record_test = 'record/%s/tsrate_%s_batch_tra_%s_batch_tes_%s_lr_%s_seed_%s_%s_test.txt' % (
    record_name, args.ts_rate, args.batch_size_tra, args.batch_size_tes, args.lr, args.seed, record_num)
    # record_pro = 'record/%s/batch_tra_%s_batch_tes_%s_lr_%s_seed_%s_%s_probability.txt' % (record_name, args.batch_size_tra, args.batch_size_tes, args.lr, args.seed, record_num)
    while os.path.exists(record_train):
        record_num += 1
        record_train = 'record/%s/tsrate_%s_batch_tra_%s_batch_tes_%s_lr_%s_seed_%s_%s.txt' % (
        record_name, args.ts_rate, args.batch_size_tra, args.batch_size_tes, args.lr, args.seed, record_num)
        record_test = 'record/%s/tsrate_%s_batch_tra_%s_batch_tes_%s_lr_%s_seed_%s_%s_test.txt' % (
        record_name, args.ts_rate, args.batch_size_tra, args.batch_size_tes, args.lr, args.seed, record_num)
        # record_pro = 'record/%s/batch_tra_%s_batch_tes_%s_lr_%s_seed_%s_%s_probability.txt' % (record_name, args.batch_size_tra, args.batch_size_tes, args.lr, args.seed, record_num)

    with open(record_train, 'a') as record:
        record.write(
            '--source: %s\n--target: %s\n--model: %s\n--ts_rate: %s\n--rate: %s\n--num_k: %s\n--max_epoch: %s\n\n' % (
            args.source, args.target, args.model,
            args.ts_rate, args.rate, args.num_k, args.max_epoch))
    with open(record_test, 'a') as record:
        record.write(
            '--source: %s\n--target: %s\n--model: %s\n--ts_rate: %s\n--rate: %s\n--num_k: %s\n--max_epoch: %s\n\n' % (
            args.source, args.target, args.model,
            args.ts_rate, args.rate, args.num_k, args.max_epoch))
    solver = Solver(args, source=args.source, target=args.target, model_name=args.model, num_workers=args.num_workers,
                    MRDA_dim=args.MRDA_dim, pin_memory=args.pin_memory, learning_rate=args.lr, batch_size_tra=args.batch_size_tra,
                    batch_size_tes=args.batch_size_tes, num_k=args.num_k, save_epoch=args.save_epoch,
                    record_train=record_train, record_test=record_test, tlabel=args.tlabel, seed=args.seed, lb_rate=args.lb_rate,
                    ts_rate=args.ts_rate, gama=args.gama, MFFA_pt=args.MFFA_pt, MRDA_pt=args.MRDA_pt, expl=args.expl)
    for t in range(args.max_epoch):
        num = solver.train(t, record_file=record_train)
        F1 = solver.test(t, record_file=record_test)
        list_F1s.append(F1)
        time_end = time.time()
        with open(record_test, 'a') as record:
            record.write('Epoch: %s\t Totally cost: %s\n' % (t, time_end - time_start))
    with open(record_test, 'a') as record:
        record.write('Max accuracy epoch: %s\n' % (list_F1s.index(max(F1))))
    print('Max accuracy epoch: %s\t Totally cost: %s' % (list_F1s.index(max(F1)), time_end - time_start))

if __name__ == '__main__':
    main()
