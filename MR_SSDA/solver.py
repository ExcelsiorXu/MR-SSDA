from __future__ import print_function
import sys, os
import numpy as np

sys.path.append('./datasets')
from load_data import UnalignedDataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from model.build_net import Generator, Classifier


# from sklearn.externals import joblib

# Training settings
class Solver(object):
    def __init__(self, args, model_name='none', batch_size_tra=32, batch_size_tes=1, source='230ICIAR-2018', target='230BreaKHis200',
                 MRDA_dim=512, num_workers=4, pin_memory=True, learning_rate=0.0001, interval=80, num_k=4,
                 all_use=False, checkpoint_dir=None, save_epoch=5, record_train='train', record_test='test',
                 seed=1, rate=0, ts_rate=0, MRDA_pt=0, MFFA_pt=0, gama=0.9, expl=None):
        self.batch_size = batch_size_tra
        self.batch_size_tes = batch_size_tes
        self.source = source
        self.target = target
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_k = num_k
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        self.record_train = record_train
        self.record_test = record_test
        self.MRDA_dim = MRDA_dim
        self.seed = seed
        self.rate = rate
        self.ts_rate = ts_rate
        self.model = model_name
        self.MFFA_pt = MFFA_pt
        self.MRDA_pt = MRDA_pt
        self.gama = gama
        # self.model_name = model_name
        self.expl = expl

        if self.source == 'svhn':
            self.scale = True
        else:
            self.scale = False
        print('dataset loading...')

        S_train = './dataset/' + self.source + '/train/train_all'
        train_loader = UnalignedDataLoader()
        self.dataset_S_train, self.S_train_num = train_loader.initialize(S_train, self.batch_size, True,
                                                                         self.num_workers,
                                                                         self.pin_memory)

        T_train = './dataset/' + self.target + '/train/train'
        self.dataset_T_train, self.T_train_num = train_loader.initialize(T_train, self.batch_size, True,
                                                                         self.num_workers,
                                                                         self.pin_memory)

        T_train_label = './dataset/' + self.target + '/train/train_label'
        train_loader = UnalignedDataLoader()
        self.dataset_T_train_label, self.T_train_label_num = train_loader.initialize(T_train_label, self.batch_size,
                                                                                     True,
                                                                                     self.num_workers, self.pin_memory)
        print('load finished!\n')

        self.G = Generator(model=self.model)
        print("self.G: ", self.G)
        with open(record_train, 'a') as record:
            record.write('self.G: %s\n' % (self.G))
        self.C = Classifier(model=self.model)
        print("self.C: ", self.C)

        with open(record_train, 'a') as record:
            record.write('self.C: %s\n' % (self.C))
        with open(record_train, 'a') as record:
            record.write('--source_train_number: %s\n--target_train_number: %s\n--target_train_label_number: %s\n\n' % (
                self.S_train_num, self.T_train_num, self.T_train_label_num))

        print("self.model: ", self.model)
        with open(record_train, 'a') as record:
            record.write('self.model: %s\n' % (self.model))
        self.G.cuda()
        self.C.cuda()

        self.interval = interval
        self.lr = learning_rate

    def train(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C.train()
        correct1 = 0
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        t_num = 0
        t_label_num = 0

        if epoch == 0:
            self.lr_last = self.lr
        opt_G = optim.Adam(self.G.parameters(), lr=self.lr_last)
        opt_C = optim.Adam(self.C.parameters(), lr=self.lr_last)
        scheduler = lr_scheduler.ExponentialLR(opt_G, gamma=self.gama)

        for batch_idx, data in enumerate(self.dataset_S_train):
            if (batch_idx + 1) * self.batch_size > self.S_train_num * self.ts_rate:
                break

            if (batch_idx + 1) % self.rate == 0:
                for batch_idx_T, data_T in enumerate(self.dataset_T_train_label, t_label_num):
                    t_label_num += 1
                    img_s = data['data']
                    img_t = data_T['data'] 
                    label_s = data['label']
                    label_t = data_T['label']
                    if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                        break

                    img_s = Variable(img_s.cuda())
                    img_t = Variable(img_t.cuda())
                    label_s = Variable(label_s.long().cuda())
                    label_t = Variable(label_t.long().cuda())

                    opt_G.zero_grad()
                    opt_C.zero_grad()

                    feat_s = self.G(img_s)
                    feat_t = self.G(img_t)
                    output_s1, s_bn1, s_x12, s_x12_MLFF, s_bn2 = self.C(feat_s)
                    output_t1, t_bn1, t_x12, t_x12_MLFF, t_bn2 = self.C(feat_t)

                    pred1 = output_s1.data.max(1)[1]
                    correct1 += pred1.eq(label_s.data).cpu().sum()

                    loss_s = criterion(output_s1, label_s)
                    MRDA_loss_tar = MRDA_loss(t_x12, t_x12_MLFF, 10)  # beta = 10
                    MRDA_loss_src = MRDA_loss(s_x12, s_x12_MLFF, 10)  # beta = 10
                    MFFA_loss_bn1 = MFFA_rbf(s_bn1, t_bn1)
                    MFFA_loss_bn2 = MFFA_rbf(s_bn2, t_bn2)

                    loss = loss_s + self.MFFA_pt * (MFFA_loss_bn1 + MFFA_loss_bn2) + self.MRDA_pt * (MRDA_loss_tar + MRDA_loss_src)
                    loss_t = criterion(output_t1, label_t)
                    loss += loss_t

                    batch_cnt = batch_idx + 1
                    img_select = self.batch_size * batch_cnt
                    loss.backward()
                    opt_G.step()
                    opt_C.step()

                    if batch_cnt % self.interval == 0:
                        # print('=================================' + str(t_label_num) + '===========================================')
                        print(
                            'Train Epoch: {} [{}/{} ({:.2f}%)]\t Accuracy: {}/{} ({:.6f}%)\t cross_Loss: {:.6f}\t'.format(
                                epoch, img_select, self.S_train_num * self.ts_rate,
                                                   img_select / (self.S_train_num * self.ts_rate) * 100,
                                correct1, img_select, float(correct1) / img_select * 100,
                                loss_s.item()))
                        if record_file:
                            record = open(record_file, 'a')
                            record.write('Train Epoch: %s\t Accuracy: %.6f\t cross_Loss: %.6f\t\n' % (epoch,
                                                                                                      float(
                                                                                                          correct1) / img_select,
                                                                                                      loss_s.item()))
                            record.close()
                    break
            else:
                for batch_idx_T, data_T in enumerate(self.dataset_T_train, t_num):
                    t_num += 1
                    img_t = data_T['data']
                    img_s = data['data']
                    label_s = data['label']
                    if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                        break
                    img_s = Variable(img_s.cuda())
                    img_t = Variable(img_t.cuda())
                    label_s = Variable(label_s.long().cuda())

                    opt_G.zero_grad()
                    opt_C.zero_grad()

                    feat_s = self.G(img_s)
                    feat_t = self.G(img_t)
                    output_s1, s_bn1, s_x12, s_x12_MLFF, s_bn2 = self.C(feat_s)
                    output_t1, t_bn1, t_x12, t_x12_MLFF, t_bn2 = self.C(feat_t)

                    pred1 = output_s1.data.max(1)[1]
                    correct1 += pred1.eq(label_s.data).cpu().sum()

                    loss_s = criterion(output_s1, label_s)
                    MRDA_loss_tar = MRDA_loss(t_x12, t_x12_MLFF, 10)  # beta = 10
                    MRDA_loss_src = MRDA_loss(s_x12, s_x12_MLFF, 10)  # beta = 10
                    MFFA_loss_bn1 = MFFA_rbf(s_bn1, t_bn1)
                    MFFA_loss_bn2 = MFFA_rbf(s_bn2, t_bn2)

                    loss = loss_s + self.MFFA_pt * (MFFA_loss_bn1 + MFFA_loss_bn2) + self.MRDA_pt * (MRDA_loss_tar + MRDA_loss_src)

                    batch_cnt = batch_idx + 1
                    img_select = self.batch_size * batch_cnt
                    loss.backward()
                    opt_G.step()
                    opt_C.step()

                    if batch_cnt % self.interval == 0:
                        print(
                            'Train Epoch: {} [{}/{} ({:.2f}%)]\t Accuracy: {}/{} ({:.6f}%)\t cross_Loss: {:.6f}\t'.format(
                                epoch, img_select, self.S_train_num * self.ts_rate,
                                                   img_select / (self.S_train_num * self.ts_rate) * 100,
                                correct1, img_select, float(correct1) / img_select * 100,
                                loss_s.item()))
                        if record_file:
                            record = open(record_file, 'a')
                            record.write('Train Epoch: %s\t Accuracy: %.6f\t cross_Loss: %.6f\t\n' % (epoch,
                                                                                                      float(
                                                                                                          correct1) / img_select,
                                                                                                      loss_s.item()))
                            record.close()
                    break

        print(epoch, 'lr={:.10f}'.format(scheduler.get_lr()[0]))
        scheduler.step()
        self.lr_last = scheduler.get_lr()[0]
        print('train_batch', batch_idx + 1)

        PKL_DIR = self.source + '_' + self.target + '_' + self.model + '_' + self.expl
        PKL_DIR = 'PKL/' + PKL_DIR
        if not os.path.exists(PKL_DIR):
            os.mkdir(PKL_DIR)
        if epoch == self.save_epoch and self.source == '230ICIAR-2018':
            torch.save(self.G.state_dict(), PKL_DIR + '/IBG_m.pkl')
            torch.save(self.C.state_dict(), PKL_DIR + '/IBC_m.pkl')
        if epoch == self.save_epoch and self.source == '230BreaKHis200':
            torch.save(self.G.state_dict(), PKL_DIR + '/BIG_m.pkl')
            torch.save(self.C.state_dict(), PKL_DIR + '/BIC_m.pkl')

        return 0

    def test(self, epoch, record_file=None):
        self.G.eval()
        self.C.eval()
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        list_acc = []

        T_val = './dataset/' + self.target + '/test'  # 目标域

        for file_test in os.listdir(T_val):
            label_dict = {}
            list1 = []
            T_test = T_val + '/' + file_test
            test_loader = UnalignedDataLoader()
            self.dataset_V_train, self.V_train_num = test_loader.initialize(T_test, self.batch_size_tes, True,
                                                                            self.num_workers,
                                                                            self.pin_memory)

            for batch_idx_V, data_V in enumerate(self.dataset_V_train):
                img = data_V['data']
                img_file_label = data_V['label']
                label = file_test
                img, img_file_label = img.cuda(), img_file_label.long().cuda()
                feat_tt = self.G(img)
                output1, t_bn1, t_x12, t_x12_MRDA, t_bn2 = self.C(feat_tt)
                pred1 = output1.data  # .max(1)[1].cpu()
                pred1 = pred1.cpu().numpy()
                img_file_label = img_file_label.cpu().numpy()
                pre_num = 0
                for key_name in img_file_label:
                    if key_name not in label_dict.keys():
                        label_dict[key_name] = pred1[pre_num]
                    else:
                        label_dict[key_name] += pred1[pre_num]
                    pre_num += 1
            for key in label_dict.keys():
                bb = label_dict[key]
                list1.append(bb.tolist())
                y_predict = np.argmax(bb)
                y_actual = int(label)
                aa = 1 if y_predict == y_actual else 0
                list_acc.append(aa)

                if y_actual == y_predict == 1:
                    TP += 1
                if y_predict == 1 and y_actual != y_predict:
                    FP += 1
                if y_actual == y_predict == 0:
                    TN += 1
                if y_predict == 0 and y_actual != y_predict:
                    FN += 1
        print('TP', TP, 'FP', FP, 'TN', TN, 'FN', FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (FP + TN)
        precision = TP / (TP + FP)
        recall = sensitivity
        F1_Score = 2 * (precision * recall) / (precision + recall)
        acc = float(TN + TP) / (TN + TP + FN + FP)

        print('\nTest Epoch: {}\t F1_Score: {}\t Accuracy: {}/{} ({:.6f}%)\t sensitivity: {}\t specificity: {}'.format(
            epoch, F1_Score, (TN + TP), (TN + TP + FN + FP), acc,
            sensitivity,
            specificity))

        if record_file:
            record = open(record_file, 'a')
            print('recording: ', record_file, '\n')
            record.write('Test Epoch: %s\t' % (epoch))
            record.write('F1_Score: %.6f\t Accuracy: %.6f\t sensitivity: %.6f\t specificity: %.6f\n' % (
                F1_Score, acc, sensitivity, specificity))
            record.write('TP: %s\t FP: %s\t TN: %s\t FN: %s\t ' % (TP, FP, TN, FN))
            record.close()
        return F1_Score


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def MFFA_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def MRDA_loss(x, x_MLFF, beta):
    R = 0
    x = x.reshape(x.size(0), x.size(1))
    for i in range(x.size(0)):
        for j in range(x.size(0)):
            s_ij = torch.exp(-(torch.norm(x[i] - x[j]) ** 2) / beta)
            r_ij = torch.norm(x_MLFF[i] - x_MLFF[j]) ** 2
            R += 1 / 2 * r_ij * s_ij
    return R / x.size(0)


if __name__ == '__main__':
    code = 0
