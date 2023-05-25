import os
import time
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import pandas as pd
import matplotlib.pyplot as plt

from model.cgcnn.data_process import CIFData
from model.cgcnn.data_process import collate_pool, get_train_val_test_loader
from model.cgcnn.model import CrystalGraphConvNet


class Crystal_Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.load_data()
        self.build_model()

        np.set_printoptions(precision=4)

        torch.set_printoptions(precision=4)

    def load_data(self):
        self.dataset = CIFData(label_dir=self.cfg.label_dir, properties=self.cfg.properties,
                               struct_dir=self.cfg.structure_dir, aug=self.cfg.augmentation)
        collate_fn = collate_pool
        self.train_loader, self.test_loader = get_train_val_test_loader(
            dataset=self.dataset,
            collate_fn=collate_fn,
            batch_size=self.cfg.batch_size,
            train_ratio=self.cfg.train_ratio,
            num_workers=self.cfg.workers,
            val_ratio=self.cfg.val_ratio,
            test_ratio=self.cfg.test_ratio,
            pin_memory=self.cfg.cuda,
            train_size=self.cfg.train_size,
            val_size=self.cfg.val_size,
            test_size=self.cfg.test_size,
            return_test=True)

        # obtain target value normalizer

        sample_data_list = [self.dataset[i] for i in sample(range(len(self.dataset)), min(200, len(self.dataset)))]
        _, sample_target, _ = collate_pool(sample_data_list)
        self.normalizer = Normalizer(sample_target)

    def build_model(self):
        structures, _, _ = self.dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        self.model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                         atom_fea_len=self.cfg.atom_fea_len,
                                         n_conv=self.cfg.n_conv,
                                         h_fea_len=self.cfg.h_fea_len,
                                         n_h=self.cfg.n_h, outsize=self.cfg.out_size, dropout=self.cfg.dropout)

        if self.cfg.cuda:
            self.model.cuda()

        # define loss func and optimizer
        self.criterion = nn.MSELoss()

        if self.cfg.optim == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), self.cfg.lr,
                                       momentum=self.cfg.momentum,
                                       weight_decay=self.cfg.weight_decay)
        elif self.cfg.optim == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), self.cfg.lr,
                                        weight_decay=self.cfg.weight_decay)
        else:
            raise NameError('Only SGD or Adam is allowed as --optim')

        self.scheduler = MultiStepLR(self.optimizer, milestones=self.cfg.lr_milestones, gamma=0.1)

    def train(self):
        # global args, best_mae_error

        best_mae_error = 1e10

        # load data

        # build model

        for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
            # train for one epoch
            self.train_epoch(epoch)

        # test best model
        print('---------Evaluate Model on Test Set---------------')

        root_dir = os.path.dirname(os.path.realpath(__file__))
        save_dir = os.path.join(root_dir, "save_dir", self.cfg.save_dir, )
        save_path = os.path.join(save_dir, "model.pth")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_checkpoint({
            'state_dict': self.model.state_dict(),
            'normalizer': self.normalizer.state_dict(),
            'args': vars(self.cfg)
        }, filename=save_path)

        self.validate(self.test_loader, test=True)

    def train_epoch(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        mae_errors = AverageMeter()

        # switch to train mode
        self.model.train()
        end = time.time()

        for i, (input, target, _) in enumerate(self.train_loader):
            # measure data loading time

            data_time.update(time.time() - end)

            if self.cfg.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[4]])

            # normalize target

            target_normed = self.normalizer.norm(target)

            target_var = Variable(target_normed.cuda(non_blocking=True))

            # compute output
            output = self.model(*input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss

            mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))

            # compute gradient and do SGD step

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if epoch % self.cfg.print_freq == self.cfg.print_freq - 1:
            print('Epoch: [{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'MAE {mae_errors.avg}'.format(
                epoch + 1, batch_time=batch_time,
                data_time=data_time, loss=losses, mae_errors=mae_errors))

        if epoch % self.cfg.test_freq == self.cfg.test_freq - 1:
            self.validate(self.test_loader)

    def validate(self, data_loader=None):
        eval_times = 50

        if data_loader is None:
            data_loader = self.test_loader

        # switch to evaluate mode
        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        mae_error = 0
        pred_list = []
        true_list = []
        std_list = []
        correct_list = []

        with torch.no_grad():

            for i, (input, target, batch_cif_ids) in enumerate(data_loader):
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[4]])

                target_normed = self.normalizer.norm(target)

                target_var = Variable(target_normed.cuda(non_blocking=True))

                outputs = np.zeros((eval_times, len(target), len(self.cfg.properties)))
                for j in range(eval_times):
                    output = self.model(*input_var)
                    output = self.normalizer.denorm(output.data.cpu())
                    outputs[j, :] = output

                outputs = -outputs[:, :, 0] + outputs[:, :, 1]
                target = -target[:, 0] + target[:, 1]

                mean = np.mean(outputs, axis=0)
                std = np.std(outputs, axis=0)
                std_list.append(std)

                correct = np.logical_and(target.numpy() > (mean - 2 * std),
                                         target.numpy() < (mean + 2 * std))
                correct_list.append(correct)
                # print(batch_cif_ids, mean)

                # measure accuracy and record loss
                mae_error = mae_error + mae(mean, target)

            # if i % self.cfg.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
            #         i, len(data_loader), batch_time=batch_time, loss=losses,
            #         mae_errors=mae_errors))
        correct_list = np.hstack(std_list)
        percent_in_range = np.sum(correct_list, axis=0) / len(correct_list)
        std_list = np.hstack(std_list)
        std = np.mean(std_list, axis=0)

        star_label = '*'

        print(' {star} MAE {mae_errors} Percent in range {p}% STD {std}'.format(star=star_label,
                                                                                mae_errors=mae_error / (i + 1),
                                                                                std=std,
                                                                                p=percent_in_range * 100))
        return mae_error

    def predict(self, eval_times=50):

        if os.path.isfile(self.cfg.model_path):
            print("=> loading checkpoint '{}'".format(self.cfg.model_path))
            checkpoint = torch.load(self.cfg.model_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.normalizer.load_state_dict(checkpoint['normalizer'])
        else:
            print("=> no checkpoint found at '{}'".format(self.cfg.model_path))
            return 0

        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        predictions = pd.DataFrame(data=None)

        L = len(self.test_loader)
        print("Start prediction:\n")

        with torch.no_grad():
            for i, (input, target, batch_cif_ids) in enumerate(self.test_loader):

                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[4]])

                outputs = np.zeros((eval_times, len(target), len(self.cfg.properties)))
                for j in range(eval_times):
                    output = self.model(*input_var)
                    output = self.normalizer.denorm(output.data.cpu())
                    outputs[j, :] = output

                values_mean = np.mean(outputs, axis=0)
                values_std = np.std(outputs, axis=0)

                pred = {"Crystals": batch_cif_ids}

                for n, p in enumerate(self.cfg.properties):
                    pred["v_{}_mean".format(p)] = values_mean[:, n]
                    pred["v_{}_std".format(p)] = values_std[:, n]

                pred = pd.DataFrame(data=pred)

                predictions = predictions.append(pred)

                print("Progress {}/{} {:.2f}%".format((i + 1) * self.cfg.batch_size, L * self.cfg.batch_size,
                                                      (i + 1) * 100 / L))

        print("=> prediction completed")

        save_dir = os.path.dirname(self.cfg.model_path)

        save_path = os.path.join(save_dir, "predictions.csv")
        predictions.to_csv(save_path, header=True, index=False)

        return predictions

    def vis_recomm(self, preds):

        # -----------------visualization----------------#

        mean = preds["Values_mean"]
        std = preds["Values_std"]
        ub = mean + std
        lb = mean - std

        plt.plot(ub, c="cyan")
        plt.plot(lb, c="cyan", label="range")
        plt.plot(mean, c="black", label="mean", linewidth=1)

        plt.xlabel("Catalyst Index")
        plt.ylabel("Predicted Value")
        plt.title("Catalyst Property")
        plt.legend()
        plt.show()

        # -----------------recommendation----------------#

        large_var = preds.nlargest(10, 'Values_std')['Crystals']
        large_mean = preds.nlargest(10, 'Values_mean')['Crystals']

        print("""\nRecommendations:
              ---High Uncertainty---
              {}\n
              ---High Predicted Value---
              {}\n
              """.format(large_var.to_string(),
                         large_mean.to_string()))

        return 0


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor, dim=0)
        self.std = torch.std(tensor, dim=0)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """

    dif = torch.abs(target - prediction)

    return torch.mean(dif, dim=0)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
