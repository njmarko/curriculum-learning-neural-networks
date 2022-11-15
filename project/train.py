# TODO: Log experiment results with wandb
# TODO: Save models and results in experiments folder
# TODO: Create argparser for all parameters that can be defined
# TODO: Add parallelized training and logging for all experiments
# TODO: Try out model that Bengio used in his paper
# TODO: Check out some of the best modern practices for training NNs
#  https://wandb.ai/site/articles/debugging-neural-networks-with-pytorch-and-w-b-using-gradients-and-visualizations
#  https://wandb.ai/wandb-smle/integration_best_practices/reports/W-B-Integration-Best-Practices--VmlldzoyMzc5MTI2
# TODO: Fix error that appears at the start
#  wandb: ERROR Failed to sample metric: Not Supported
# TODO: Fix warning that probably happens for AUROC
#  ....\project\venv\lib\site-packages\torchmetrics\utilities\prints.py:36:
#  UserWarning: No positive samples in targets, true positive value should be meaningless.
#  Returning zero tensor in true positive score
#   warnings.warn(*args, **kwargs)
# TODO: Check why ROC curve displays class 0 as a straight line
# TODO: Consider changing x axis to epochs in wandb
# TODO: Useful resource for wandb
#  https://www.kaggle.com/code/ayuraj/experiment-tracking-with-weights-and-biases?scriptVersionId=63334832&cellId=18
import argparse
import os
import time
import timeit
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import wandb
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, roc_auc_score
from torch import optim

from data.data_loader import load_dataset
from models.cnn_v1 import CnnV1
from itertools import repeat, cycle, islice
import torch.multiprocessing as mp

from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassAUROC, \
    MulticlassConfusionMatrix
from torchmetrics import MetricCollection
import matplotlib.pyplot as plt


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(20, 20))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        img = image_batch[n]
        plt.imshow(cv2.cvtColor(img.squeeze().numpy(), cv2.COLOR_GRAY2RGB))
        label = label_batch[n].numpy()
        plt.title(label)
        plt.axis('off')
    plt.show()


def train(model, optimizer, data_loader, opt, scheduler=None):
    model.train()
    total_samples = len(data_loader.dataset)

    global_target = np.array([])
    global_pred = np.array([])
    global_probs = np.empty((0, 3))

    running_loss = 0.0

    metrics = MetricCollection({'train_f1_micro': MulticlassF1Score(num_classes=opt.num_classes, average='micro'),
                                'train_f1_macro': MulticlassF1Score(num_classes=opt.num_classes, average='macro'),
                                'train_precision': MulticlassPrecision(num_classes=opt.num_classes),
                                'train_recall': MulticlassRecall(num_classes=opt.num_classes),
                                }
                               )
    auroc = MulticlassAUROC(num_classes=opt.num_classes, average='macro')

    start_time = timeit.default_timer()
    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        predictions = model(data.to(opt.device))
        probs = F.softmax(predictions, dim=1)
        _, pred = torch.max(probs, dim=1)
        target = target.to(opt.device)

        loss = F.nll_loss(probs, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        target = target.cpu().detach()
        pred = pred.cpu().detach()
        probs = probs.cpu().detach()
        metrics(pred, target)
        auroc(probs, target)
        global_target = np.concatenate((global_target, target.numpy()))
        global_pred = np.concatenate((global_pred, pred.numpy()))
        global_probs = np.vstack((global_probs, probs.numpy()))

        running_loss += loss.item() * data.size(0)

        if i % 5 == 0:
            wandb.log({"train_lr": scheduler.get_last_lr()[0]},
                      # commit=False, # Commit=False just accumulates data
                      )

    epoch_loss = running_loss / total_samples

    epoch_time = timeit.default_timer() - start_time
    print(f"Epoch time {epoch_time}")
    log_metrics = {
        **metrics.compute(),
        "train_epoch_loss": epoch_loss,
        "train_epoch_time": epoch_time,
        "train_confusion_matrix": wandb.plot.confusion_matrix(probs=global_probs, y_true=global_target,
                                                              class_names=['ellipse', 'square', 'triangle'],
                                                              title="Train confusion matrix"),
        "train_roc": wandb.plot.roc_curve(y_true=global_target, y_probas=global_probs,
                                          labels=['ellipse', 'square', 'triangle'],
                                          # TODO: Deterimn why classes_to_plot doesn't work with roc
                                          # classes_to_plot=['ellipse', 'square', 'triangle'],
                                          title="Train ROC", ),
        "train_auroc_macro": auroc.compute()
    }
    return log_metrics


def validation(model, data_loader, opt):
    model.eval()

    total_samples = len(data_loader.dataset)
    # correct_samples = 0

    global_target = np.array([])
    global_pred = np.array([])
    global_probs = np.empty((0, 3))

    running_loss = 0.0

    metrics = MetricCollection({'val_f1_micro': MulticlassF1Score(num_classes=opt.num_classes, average='micro'),
                                'val_f1_macro': MulticlassF1Score(num_classes=opt.num_classes, average='macro'),
                                'val_precision': MulticlassPrecision(num_classes=opt.num_classes),
                                'val_recall': MulticlassRecall(num_classes=opt.num_classes),
                                }
                               )
    auroc = MulticlassAUROC(num_classes=opt.num_classes, average='macro')
    start_time = timeit.default_timer()
    with torch.no_grad():
        for data, target in data_loader:
            res = model(data.to(opt.device))
            probs = F.softmax(res, dim=1)
            target = target.to(opt.device)
            probs = probs.to(opt.device)
            loss = F.nll_loss(probs, target, reduction='sum')
            _, pred = torch.max(probs, dim=1)

            target = target.cpu().detach()
            pred = pred.cpu().detach()
            probs = probs.cpu().detach()
            metrics(pred, target)
            auroc(probs, target)
            global_target = np.concatenate((global_target, target.numpy()))
            global_pred = np.concatenate((global_pred, pred.numpy()))
            global_probs = np.vstack((global_probs, probs.numpy()))

            running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / total_samples

    epoch_time = timeit.default_timer() - start_time

    # TODO: Wrongly classified images can also be logged with wandb
    #  https://docs.wandb.ai/ref/python/log#image-from-numpy
    log_metrics = {
        **metrics.compute(),
        "val_epoch_loss": epoch_loss,
        "val_evaluation_time": epoch_time,
        "val_confusion_matrix": wandb.plot.confusion_matrix(probs=global_probs, y_true=global_target,
                                                            class_names=['ellipse', 'square', 'triangle'],
                                                            title="Validation confusion matrix"),
        "val_roc": wandb.plot.roc_curve(y_true=global_target, y_probas=global_probs,
                                        labels=['ellipse', 'square', 'triangle'],
                                        # classes_to_plot=['ellipse', 'square', 'triangle'],
                                        title="Validation ROC", ),
        "val_auroc_macro": auroc.compute(),
    }
    return log_metrics


def create_arg_parser(model_choices=None, optimizer_choices=None, scheduler_choices=None):
    # Default values for choices
    if scheduler_choices is None:
        scheduler_choices = {'cycliclr': optim.lr_scheduler.CyclicLR}
    if optimizer_choices is None:
        optimizer_choices = {'adamw': optim.AdamW}
    if model_choices is None:
        model_choices = {'cnnv1': CnnV1}

    parser = argparse.ArgumentParser()
    # Wandb logging options
    parser.add_argument('-entity', '--entity', type=str, default="weird-ai-yankovic",
                        help="Name of the team. Multiple projects can exist for the same team.")
    parser.add_argument('-project_name', '--project_name', type=str, default="curriculum_learning",
                        help="Name of the project. Each experiment in the project will be logged separately"
                             " as a group")
    parser.add_argument('-group', '--group', type=str, default="default_experiment",
                        help="Name of the experiment group. Each model in the experiment group will be logged "
                             "separately under a different type.")
    parser.add_argument('-save_model_wandb', '--save_model_wandb', type=bool, default=True,
                        help="Save best model to wandb run.")

    # Dataset options
    parser.add_argument('-d', '--dataset', type=str, default="data/generated_images/dataset3",
                        help="Path to the dataset")
    parser.add_argument('-b', '--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('-shuffle', '--shuffle', type=bool, default=True, help="Shuffle dataset")
    parser.add_argument('-nw', '--num_workers', type=int, default=0, help="Number of workers to be used")
    parser.add_argument('-nc', '--num_classes', type=int, default=3, help="Number of classes that can be detected")
    parser.add_argument('-ts', '--training_split', type=float, default=0.8, help="Train split between 0 and 1")
    parser.add_argument('-vs', '--validation_split', type=float, default=0.1, help="Validation split between 0 and 1")
    parser.add_argument('-es', '--evaluation_split', type=float, default=0.1,
                        help="Evaluation (test) split between 0 and 1")

    # Model options
    parser.add_argument('-m', '--model', type=str.lower, default=CnnV1.__name__,
                        choices=model_choices.keys(),
                        help=f"Model to be used for training {model_choices.keys()}")
    parser.add_argument('-depth', '--depth', type=int, default=2, help="Model depth")
    parser.add_argument('-in_channels', '--in_channels', type=int, default=1, help="Number of in channels")
    parser.add_argument('-out_channels', '--out_channels', type=int, default=8, help="Number of out channels")
    parser.add_argument('-kernel_dim', '--kernel_dim', type=int, default=3,
                        help="Kernel dimension used by CNN models")
    parser.add_argument('-mlp_dim', '--mlp_dim', type=int, default=3,
                        help="Dimension of mlp at the end of the model. Should be the same as the number of classes")
    parser.add_argument('-padding', '--padding', type=int, default=1, help="Padding used by CNN models")
    parser.add_argument('-stride', '--stride', type=int, default=1, help="Stride used by CNN models")
    parser.add_argument('-max_pool', '--max_pool', type=int, default=3, help="Max pool dimensions used by CNN models")
    parser.add_argument('-dropout', '--dropout', type=float, default=0.2, help="Dropout used in models")

    # Training options
    parser.add_argument('-device', '--device', type=str, default='cuda', help="Device to be used")
    parser.add_argument('-e', '--n_epochs', type=int, default=20, help="Number of epochs")
    parser.add_argument('-exp_name', '--exp_name', type=str, default="default_experiment",
                        help="Name of the experiment")
    parser.add_argument('-nm', '--n_models', type=int, default=50, help="Number of models to be trained")
    parser.add_argument('-pp', '--parallel_processes', type=int, default=0,
                        help="Number of parallel processes to spawn for models [0 for all available cores]")

    # Optimizer options
    parser.add_argument('-optim', '--optimizer', type=str.lower, default="adamw",
                        choices=optimizer_choices.keys(),
                        help=f'Optimizer to be used {optimizer_choices.keys()}')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.05, help="Weight decay for optimizer")

    # Scheduler options
    parser.add_argument('-sch', '--scheduler', type=str.lower, default='cycliclr',
                        choices=scheduler_choices.keys(),
                        help=f'Optimizer to be used {scheduler_choices.keys()}')
    parser.add_argument('-base_lr', '--base_lr', type=float, default=3e-4,
                        help="Base learning rate for scheduler")
    parser.add_argument('-max_lr', '--max_lr', type=float, default=0.001,
                        help="Max learning rate for scheduler")
    parser.add_argument('-step_size_up', '--step_size_up', type=int, default=0,
                        help="CycleLR scheduler: step size up. If 0, then it is automatically calculated.")
    parser.add_argument('-cyc_mom', '--cycle_momentum', type=bool, default=False,
                        help="CyclicLR scheduler: cycle momentum in scheduler")
    parser.add_argument('-sch_m', '--scheduler_mode', type=str, default="triangular2",
                        choices=['triangular', 'triangular2', 'exp_range'],
                        help=f"CyclicLR scheduler: mode {['triangular', 'triangular2', 'exp_range']}")
    return parser


def find_balanced_chunk_size(lst_size, n_processes):
    chunk = lst_size // (n_processes - 1)
    # Balanced chunks (ex. list of len 50 will be split into 4 chunks of lengths [13,13,13,11] instead of [16,16,16,2]
    while lst_size % chunk < chunk and lst_size // (chunk - 1) < n_processes:
        chunk -= 1
    return chunk


def get_chunked_lists(opt):
    model_ids = [f'model_{i}' for i in range(opt.n_models)]
    if opt.parallel_processes == 0:
        chunk = len(model_ids) // (mp.cpu_count() - 1)
    else:
        chunk = len(model_ids) // (opt.parallel_processes - 1)
        # Balanced chunks
        while len(model_ids) % chunk < chunk and len(model_ids) // (chunk - 1) < opt.parallel_processes:
            chunk -= 1
    epochs = [e for e in range(10, opt.n_epochs)]
    epoch_ranges = list(islice(cycle(epochs), opt.n_models))
    epoch_splits = [epoch_ranges[i:i + chunk] for i in range(0, len(epoch_ranges), chunk)]
    model_id_splits = [model_ids[i:i + chunk] for i in range(0, len(model_ids), chunk)]
    return epoch_splits, model_id_splits


def pass_right_constructor_arguments(target_class, opt):
    # TODO: Create an instance of a class by sending only the arguments that exist in the constructor
    pass


def create_experiments():
    # TODO: Run Experiments in parallel
    # TODO: Pass specific options for each experiment
    # TODO: Determine if it is better to create all processes at the start, where each process goes through the slice
    #  of list of all epochs, or create them for each experiment where each experiment takes only one epoch value

    parser = create_arg_parser()
    opt = parser.parse_args()

    model_ids = [f'model_{i}' for i in range(opt.n_models)]
    epochs = [e for e in range(10, opt.n_epochs)]
    # TODO: Find a way to slice the epochs if number of models is lower than number of epochs.
    #  In that case, maybe every second or third epoch should be tested
    epoch_ranges = list(islice(cycle(epochs), opt.n_models))
    print(len(epoch_ranges))

    with mp.Pool(opt.parallel_processes) as pool:
        pool.starmap(run_experiment, zip(epoch_ranges, model_ids))


def run_experiment(max_epoch, model_id):
    # TODO: Make run_experiment more generic to support different types of experiments other than variable epochs ones

    # Model options
    model_choices = {CnnV1.__name__.lower(): CnnV1, }  # TODO: Add more model choices

    optimizer_choices = {optim.AdamW.__name__.lower(): optim.AdamW, }  # TODO: Add more optimizer choices

    # Scheduler options
    scheduler_choices = {
        optim.lr_scheduler.CyclicLR.__name__.lower(): optim.lr_scheduler.CyclicLR, }  # TODO: Add more scheduler choices

    parser = create_arg_parser(model_choices=model_choices, optimizer_choices=optimizer_choices,
                               scheduler_choices=scheduler_choices)
    opt = parser.parse_args()

    opt.n_epochs = max_epoch
    print(f'{model_id} is training')

    # Add specific options for experiments

    opt.device = 'cuda' if torch.cuda.is_available() and (opt.device == 'cuda') else 'cpu'
    print(opt.device)
    if opt.device == 'cuda':
        print(f'GPU {torch.cuda.get_device_name(0)}')

    # TODO: Determine if we need to fix the seed for every dataset
    # TODO: Determine how random split is created. Maybe make sure that it always takes a certain percentage of
    #  easy, medium and hard examples
    # TODO: Decide if pin_memory is worth it
    train_loader, val_loader, test_loader = load_dataset(base_dir=opt.dataset, batch_size=opt.batch_size,
                                                         lengths=[opt.training_split, opt.validation_split,
                                                                  opt.evaluation_split],
                                                         shuffle=opt.shuffle, num_workers=opt.num_workers,
                                                         pin_memory=False)

    # TODO: Determine optimal step_size_up for cyclicLR scheduler.
    if opt.step_size_up <= 0:
        opt.step_size_up = 2 * len(train_loader.dataset) // opt.batch_size

    wb_run_train = wandb.init(entity=opt.entity, project=opt.project_name, group=opt.group,
                              # save_code=True, # Pycharm complains about duplicate code fragments
                              job_type="train",
                              tags=['variable_epochs'],
                              name=f'{model_id}_train',
                              config=opt,
                              )

    # Define model
    model = model_choices[opt.model](depth=opt.depth, in_channels=opt.in_channels, out_channels=opt.out_channels,
                                     kernel_dim=opt.kernel_dim, mlp_dim=opt.mlp_dim, padding=opt.padding,
                                     stride=opt.stride, max_pool=opt.max_pool,
                                     dropout=opt.dropout)

    model = model.to(opt.device)

    # TODO: Test SGD with momentum with parameters that look similar to this
    #  optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    #  scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
    # TODO: Test LRFinder
    #  lr_finder = LRFinder(net, optimizer, device)
    #  lr_finder.range_test(trainloader, end_lr=10, num_iter=100, logwandb=True)

    optimizer = optimizer_choices[opt.optimizer](model.parameters(), lr=opt.learning_rate,
                                                 weight_decay=opt.weight_decay)

    # TODO: Scheduler worked better when base and max values were reversed. We should look into that
    #   Maybe try base_lr of 0.001 with some other value for max_lr.
    #   LRFinder could help us determine the optimal base_lr
    scheduler = scheduler_choices[opt.scheduler](optimizer=optimizer, base_lr=opt.base_lr, max_lr=opt.max_lr,
                                                 step_size_up=opt.step_size_up,
                                                 cycle_momentum=opt.cycle_momentum, mode=opt.scheduler_mode)

    # TODO: Check model gradients. Make sure gradients are not vanishing/exploding
    #  wandb.watch(net, log='all')

    best_model_f1_macro = -np.Inf
    best_model_path = None
    best_epoch = 0
    artifact = wandb.Artifact(name=f'train-{opt.exp_name}-{model_id}-max_epochs{opt.n_epochs}', type='model')
    for epoch in range(1, opt.n_epochs + 1):
        print(f"{epoch=}")
        train_metrics = train(model=model, optimizer=optimizer, data_loader=train_loader, opt=opt,
                              scheduler=scheduler)
        val_metrics = validation(model=model, data_loader=val_loader, opt=opt)

        # TODO: Add early stopping - Maybe not needed for this experiment. In that case log tables before ending
        if epoch < opt.n_epochs:
            del train_metrics["train_confusion_matrix"]
            del train_metrics["train_roc"]
            del val_metrics["val_confusion_matrix"]
            del val_metrics["val_roc"]
        wandb.log(train_metrics)
        wandb.log(val_metrics)

        # TODO: Save both best model and last model for the experiment
        #  (ex. best was in epoch 16 but last was also saved in epoch 20)
        if val_metrics['val_f1_macro'] > best_model_f1_macro:
            print(f"Saving model with new best {val_metrics['val_f1_macro']=}")
            best_model_f1_macro, best_epoch = val_metrics['val_f1_macro'], epoch
            Path(f'experiments/{opt.exp_name}').mkdir(exist_ok=True)
            new_best_path = os.path.join(f'experiments/{opt.exp_name}',
                                         f'train-{opt.exp_name}-{model_id}-max_epochs{opt.n_epochs}-epoch{epoch}'
                                         f'-metric{val_metrics["val_f1_macro"]:.4f}.pt')
            torch.save(model.state_dict(), new_best_path)
            # TODO: Best model can also be saved in wandb
            #  https://docs.wandb.ai/guides/models
            #  https://docs.wandb.ai/ref/python/artifact
            if best_model_path:
                os.remove(best_model_path)
            best_model_path = new_best_path

    if opt.save_model_wandb:
        artifact.add_file(best_model_path)
        wb_run_train.log_artifact(artifact)

    wb_run_train.finish()

    # Test loading
    wb_run_eval = wandb.init(entity=opt.entity, project=opt.project_name, group=opt.group,
                             # save_code=True, # Pycharm complains about duplicate code fragments
                             job_type="eval",
                             tags=['variable_epochs'],
                             name=f'{model_id}_eval',
                             config=opt,
                             )

    model = model_choices[opt.model](depth=opt.depth, in_channels=opt.in_channels, out_channels=opt.out_channels,
                                     kernel_dim=opt.kernel_dim, mlp_dim=opt.mlp_dim, padding=opt.padding,
                                     stride=opt.stride, max_pool=opt.max_pool,
                                     dropout=opt.dropout)
    model.load_state_dict(torch.load(best_model_path))
    model.to(opt.device)

    # TODO: Create a new helper function that returns the number of model parameters
    # TODO: Also save number of parameters for the model in the wandb config
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print(pytorch_total_params)
    res = validation(model=model, data_loader=test_loader, opt=opt)
    print(f"TEST FINISHED {res}")
    wb_run_eval.finish()


def main():
    create_experiments()


if __name__ == "__main__":
    main()
