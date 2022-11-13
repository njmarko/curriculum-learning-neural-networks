# TODO: Log experiment results with wandb
# TODO: Save models and results in experiments folder
# TODO: Create argparser for all parameters that can be defined
# TODO: Add parallelized training and logging for all experiments

import argparse
import os
import timeit
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import optim

from data.data_loader import load_dataset
from models.cnn_v1 import CnnV1


# TODO: add argrapser opt parameter instead of specific parameters
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
        scheduler.step()

        correct_samples += pred.eq(target).sum()
        target = target.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()

        f1_score_micro = f1_score(pred, target, average='macro')

        if i % 100 == 0:
            print(f"{f1_score_micro=}")
            # TODO: Add wandb logging

    print(f"Epoch time {timeit.default_timer() - start_time}")

    acc = correct_samples / total_samples * 100
    # TODO: Add wandb logging
    return acc


# TODO: add argrapser opt parameter instead of specific parameters
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

    f1 = f1_score(global_target, global_pred, average='macro')
    print(f"Validation {acc=} {f1=}")
    return acc


def main():
    # TODO: Move Argparser creation to a separate function. Pass choices as params
    # TODO: Create a new function that can run every individual experiment. This will allow easier parallelization

    parser = argparse.ArgumentParser()
    # Dataset options
    parser.add_argument('-d', '--dataset', type=str, default="data/generated_images/dataset3",
                        help="Path to the dataset")
    parser.add_argument('-b', '--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('-shuffle', '--shuffle', type=bool, default=True, help="Shuffle dataset")
    parser.add_argument('-nw', '--num_workers', type=int, default=1, help="Number of workers to be used")

    # Model options
    model_choices = {CnnV1.__name__.lower(): CnnV1, }  # TODO: Add model choices
    parser.add_argument('-m', '--model', type=str.lower, default=CnnV1.__name__,
                        choices=model_choices.keys(),
                        help=f"Model to be used for training {model_choices.keys()}")
    # Training options
    parser.add_argument('-device', '--device', type=str, default='cuda', help="Device to be used")
    parser.add_argument('-e', '--n_epochs', type=int, default=20, help="Number of epochs")
    parser.add_argument('-exp_name', '--exp_name', type=str, default="default_experiment",
                        help="Name of the experiment")
    # Optimizer options
    optimizer_choices = {optim.AdamW.__name__.lower(): optim.AdamW, }  # TODO: Add optimizer choices
    parser.add_argument('-optim', '--optimizer', type=str.lower, default="adamw",
                        choices=optimizer_choices.keys(),
                        help=f'Optimizer to be used {optimizer_choices.keys()}')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.05, help="Weight decay for optimizer")

    # Scheduler options
    scheduler_choices = {
        optim.lr_scheduler.CyclicLR.__name__.lower(): optim.lr_scheduler.CyclicLR, }  # TODO: Add scheduler choices
    parser.add_argument('-sch', '--scheduler', type=str.lower, default='cycliclr',
                        choices=scheduler_choices.keys(),
                        help=f'Optimizer to be used {scheduler_choices.keys()}')
    parser.add_argument('-base_lr', '--base_lr', type=float, default=0.001,
                        help="Base learning rate for scheduler")
    parser.add_argument('-max_lr', '--max_lr', type=float, default=3e-4,
                        help="Max learning rate for scheduler")
    parser.add_argument('-step_size_up', '--step_size_up', type=int, default=5,
                        help="CycleLR scheduler: step size up")
    parser.add_argument('-cyc_mom', '--cycle_momentum', type=bool, default=False,
                        help="CyclicLR scheduler: cycle momentum in scheduler")
    parser.add_argument('-sch_m', '--scheduler_mode', type=str, default="triangular2",
                        choices=['triangular', 'triangular2', 'exp_range'],
                        help=f"CyclicLR scheduler: mode {['triangular', 'triangular2', 'exp_range']}")
    opt = parser.parse_args()

    opt.device = 'cuda' if torch.cuda.is_available() and (opt.device == 'cuda') else 'cpu'
    print(opt.device)
    if opt.device == 'cuda':
        print(f'GPU {torch.cuda.get_device_name(0)}')

    # TODO: Log arg options in wandb

    # Define model
    model = model_choices[opt.model]()  # TODO: Add model parameters

    # TODO: Test SGD with momentum with parameters that look simmilar to this
    #  optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    #  scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)

    optimizer = optimizer_choices[opt.optimizer](model.parameters(), lr=opt.learning_rate,
                                                 weight_decay=opt.weight_decay)

    scheduler = scheduler_choices[opt.scheduler](optimizer, base_lr=opt.base_lr, max_lr=opt.max_lr,
                                                 step_size_up=opt.step_size_up,
                                                 cycle_momentum=opt.cycle_momentum, mode=opt.scheduler_mode)

    model = model.to(opt.device)

    train_loader, val_loader = load_dataset(base_dir=opt.dataset, batch_size=opt.batch_size,
                                            shuffle=opt.shuffle, num_workers=opt.num_workers)

    best_model_acc = -np.Inf
    best_model_path = None
    best_epoch = 0
    for epoch in range(opt.n_epochs):
        print(f"{epoch=}")
        train(model=model, optimizer=optimizer, data_loader=train_loader, device=opt.device, scheduler=scheduler)
        val_acc = validation(model=model, data_loader=val_loader, device=opt.device)

        # TODO: Save both best model and last model for the experiment
        #  (ex. best was in epoch 16 but last was also saved in epoch 20)
        if val_acc > best_model_acc:
            print(f"Saving model with new best {val_acc=}")
            best_model_acc, best_epoch = val_acc, epoch
            Path(f'experiments/{opt.exp_name}').mkdir(exist_ok=True)
            new_best_path = os.path.join(f'experiments/{opt.exp_name}',
                                         f'train-{opt.exp_name}-e{epoch}-metric{val_acc:.4f}.pt')
            torch.save(model.state_dict(), new_best_path)
            if best_model_path:
                os.remove(best_model_path)
            best_model_path = new_best_path

        # TODO: Add early stopping - Maybe not needed for this experiment

    # Test loading
    # model = CnnV1()
    # model.load_state_dict(torch.load("experiments/default_experiment/train-default_experiment-3-acc54.55000305175781"))
    # model.to(device)
    # model.eval()
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print(pytorch_total_params)
    # train_loader, val_loader = load_dataset(base_dir=opt.dataset, batch_size=opt.batch_size,
    #                                         shuffle=opt.shuffle, num_workers=opt.num_workers)
    # res = validation(model=model, data_loader=val_loader, device=device)
    # print(res)


if __name__ == "__main__":
    main()
