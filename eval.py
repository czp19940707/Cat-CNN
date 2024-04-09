# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np

from dataProc import dataProc
from torch.utils.data import DataLoader
import torch
from utils import model_select
import csv
import torch.nn.functional as F
from sklearn.metrics import roc_curve, confusion_matrix, auc, recall_score


def function1(matrix):
    return matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1],


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=2023, help='seed')
    parser.add_argument('-g', type=str, default='sMCI_pMCI', help='group')
    parser.add_argument('-b', type=int, default=16, help='batch size')
    parser.add_argument('-v', type=str, default='z15', help='net name')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model_select(args).to(device)
    # metric_result_path = os.path.join('metric_result', '{}_{}'.format(args.net, args.group))
    weights_path = r'weights'

    prob_list_sum = []
    pred_list_sum = []
    cls_list_sum = []
    metric_result_path = os.path.join('results', args.g, args.v)
    if not os.path.exists(metric_result_path):
        os.makedirs(metric_result_path)

    for fold in range(5):
        val_set = dataProc(stage='val', fold=fold, seed=args.seed, group=args.g, version=args.v)
        val_loader = DataLoader(val_set, shuffle=False, num_workers=8, batch_size=args.b)

        net.load_state_dict(
            torch.load(os.path.join(weights_path, r'{}/{}/{}.pth'.format(args.v, args.g, fold))))

        prob_list = []
        pred_list = []
        cls_list = []

        net.eval()
        for data, cls in val_loader:
            # data = data.to(device).float()
            cls = cls.to(device)
            with torch.no_grad():
                out = net.forward(data)
            probs = F.softmax(out, dim=1)  # [:, 1]
            _, perds = out.max(1)

            cls_list.append(cls)
            pred_list.append(perds)
            prob_list.append(probs)
        prob_list_sum.append(torch.cat(prob_list))
        pred_list_sum.append(torch.cat(pred_list))
        cls_list_sum.append(torch.cat(cls_list))
    prob_list_sum = torch.cat(prob_list_sum).cpu().numpy()
    pred_list_sum = torch.cat(pred_list_sum).cpu().numpy()
    cls_list_sum = torch.cat(cls_list_sum).cpu().numpy()

    np.save(os.path.join(metric_result_path, 'prob.npy'), prob_list_sum)
    np.save(os.path.join(metric_result_path, 'pred.npy'), pred_list_sum)
    np.save(os.path.join(metric_result_path, 'cls.npy'), cls_list_sum)

    print('*' * 25 + args.v + ' finish!' + '*' * 25)
