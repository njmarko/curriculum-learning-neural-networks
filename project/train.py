# TODO: Add training loop
# TODO: Log experiment results with wandb
# TODO: Save models and results in experiments folder
# TODO: Create argparser for all parameters that can be defined
import argparse
import os
from pathlib import Path

import numpy as np
from torch import optim

from models.cnn_v1 import CnnV1

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import timeit
from data.data_loader import load_dataset

def train(model, optimizer, data_loader, loss_history=None, scheduler=None, device='cpu'):
    if loss_history is None:
        loss_history = []
    model.train()
    total_samples = len(data_loader.dataset)

    correct_samples = 0
    start_time = timeit.default_timer()
    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        predictions = model(data.to(device))
        probs = F.log_softmax(predictions, dim=1)
        probs, target = probs.to(device), target.to(device)

        _, pred = torch.max(probs, dim=1)

        loss = F.nll_loss(probs, target)
        loss.backward()
        optimizer.step()

        correct_samples += pred.eq(target).sum()
        target = target.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()

        f1_score_micro = f1_score(pred, target, average='micro')

        if i % 100 == 0:
            print(f"{f1_score_micro=}")
            # TODO: Add wandb logging

    print(f"Epoch time {timeit.default_timer() - start_time}")

    acc = correct_samples / total_samples * 100
    # TODO: Add wandb logging
    return acc


def validation(model, data_loader, loss_history=None, device='cpu'):
    if loss_history is None:
        loss_history = []
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    global_target = np.array([])
    global_pred = np.array([])

    with torch.no_grad():
        for data, target in data_loader:
            res = model(data.to(device))
            res = res.to(device)
            output = F.log_softmax(res, dim=1)
            target = target.to(device)
            output = output.to(device)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

            target = target.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()

            global_target = np.concatenate((global_target, target))
            global_pred = np.concatenate((global_pred, pred))

    avg_loss = total_loss / total_samples
    acc = 100.0 * correct_samples / total_samples
    loss_history.append(avg_loss)

    f1 = f1_score(global_target, global_pred, average='micro')
    print(f"Validation {acc=} {f1=}")
    return acc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="data/generated_images/dataset5", help="Path to the dataset")
    parser.add_argument('-e', '--n_epochs', type=int, default=20)
    parser.add_argument('-exp_name', '--exp_name', type=str, default="default_experiment")
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.05)
    parser.add_argument('-optim', '--optimizer', type=str.lower,
                        choices=['adamw'],
                        help='Optimizer to be used [adamw]')  # TODO: Add optimizer choices
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-shuffle', '--shuffle', type=bool, default=True)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)

    opt = parser.parse_args()
    # TODO: Log arg options in wandb

    # Define model
    model = CnnV1()
    optimizer = optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    # TODO: Add scheduler
    model = model.to(device)

    train_loader, val_loader = load_dataset(base_dir=opt.dataset, batch_size=opt.batch_size,
                                            shuffle=opt.shuffle, num_workers=opt.num_workers)

    best_model_acc = -np.Inf
    best_epoch = 0
    for epoch in range(opt.n_epochs):
        print(f"{epoch=}")
        train(model=model, optimizer=optimizer, data_loader=train_loader, device=device)
        val_acc = validation(model=model, data_loader=val_loader, device=device)

        # if val_acc > best_model_acc:
        #     print(f"Saving model with new best {val_acc=}")
        #     best_model_acc, best_epoch = val_acc, epoch
            # Path(f'experiments/{opt.exp_name}').mkdir(exist_ok=True)
            # torch.save(model.state_dict(), os.path.join(f'experiments/{opt.exp_name}',
            #                                             f'train-{opt.exp_name}-{epoch}-acc{val_acc}'))

        # TODO: Add early stopping


if __name__ == "__main__":
    main()
