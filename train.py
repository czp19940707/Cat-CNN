from dataProc import dataProc
import argparse
import torch
import os
import time
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from utils import model_select

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=2023, help='seed')
    parser.add_argument('-fold', type=int, default=0, help='fold')
    parser.add_argument('-g', type=str, default='CN_AD', help='group')
    parser.add_argument('-b', type=int, default=8, help='batch size')
    parser.add_argument('-v', type=str, default='Cat-CNN')
    args = parser.parse_args()

    # print(args.mor)
    train_set = dataProc(stage='train', seed=args.seed, fold=args.fold, group=args.g, version=args.v)
    sample_weight, imbalanced_ratio = train_set.get_sample_weights()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
    val_set = dataProc(stage='val', seed=args.seed, fold=args.fold, group=args.g, version=args.v)

    train_loader = DataLoader(train_set, sampler=sampler, num_workers=0, batch_size=args.b, drop_last=True)
    val_loader = DataLoader(val_set, shuffle=False, num_workers=0, batch_size=args.b, drop_last=True)
    # combine_method = [i.strip() for i in open(r'selected_roi.txt', 'r')][0]
    net = model_select(args=args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    ml = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, eps=1e-8, betas=(0.9, 0.99))
    best_acc = 0.

    for epoch in range(100):
        net.train()
        start = time.time()
        for batch_index, (data, cls) in enumerate(train_loader):
            cls = cls.to(device)
            optimizer.zero_grad()

            out = net.forward(data)
            train_loss = ml(out, cls)
            train_loss.backward()
            optimizer.step()

            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'
                  .format(train_loss.item(), optimizer.param_groups[0]['lr'], epoch=epoch,
                          trained_samples=args.b * batch_index, total_samples=len(train_set))
                  )
        finish_train = time.time()

        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish_train - start))

        correct = 0.0
        net.eval()
        for data, cls in val_loader:
            # data = data.to(device).float()
            cls = cls.to(device)
            with torch.no_grad():
                out = net.forward(data)
            test_loss = ml(out, cls)
            test_loss += test_loss.item()
            _, preds = out.max(1)
            correct += preds.eq(cls).sum()
        finish_eval = time.time()
        print('Evaluating Network.....')
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(val_set),
            correct.float() / len(val_set),
            finish_eval - finish_train))
        acc = correct.float() / len(val_set)
        if acc > best_acc:
            state_save_path = os.path.join(r'weights/{}/{}'.format(args.v, args.g))
            if not os.path.exists(state_save_path):
                os.makedirs(state_save_path)
            torch.save(net.state_dict(),
                       os.path.join(state_save_path, '{}.pth'.format(args.fold)))
            best_acc = acc

