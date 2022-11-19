# TODO: Try out model that Bengio used in his paper
# TODO: Check out some of the best modern practices for training NNs
#  https://wandb.ai/site/articles/debugging-neural-networks-with-pytorch-and-w-b-using-gradients-and-visualizations
#  https://wandb.ai/wandb-smle/integration_best_practices/reports/W-B-Integration-Best-Practices--VmlldzoyMzc5MTI2
# TODO: Fix error that appears at the start
#  wandb: ERROR Failed to sample metric: Not Supported
# TODO: Check why ROC curve displays class 0 as a straight line in validation
# TODO: Consider changing x axis to epochs in wandb
# TODO: Useful resource for wandb
#  https://www.kaggle.com/code/ayuraj/experiment-tracking-with-weights-and-biases?scriptVersionId=63334832&cellId=18
# TODO: Check the amount of noise that should be added to generated images at higher difficulties. Seems too low now

import argparse
import os
import random
import re
import timeit
from itertools import cycle, islice, repeat
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import optim
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassAUROC

import wandb
from data.data_loader import load_dataset
from models.cnn_v1 import CnnV1


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(20, 20))
    for n in range(25):
        plt.subplot(5, 5, n + 1)
        img = image_batch[n]
        plt.imshow(cv2.cvtColor(img.squeeze().numpy(), cv2.COLOR_GRAY2RGB))
        label = label_batch[n].numpy()
        plt.title(label)
        plt.axis('off')
    plt.show()


def train(model, optimizer, data_loader, opt, scheduler=None):
    model.train()
    total_samples = len(data_loader.dataset)

    global_target = torch.tensor([], device=opt.device)
    global_pred = torch.tensor([], device=opt.device)
    global_probs = torch.empty((0, 3), device=opt.device)

    running_loss = 0.0

    metrics = MetricCollection({'train_f1_micro': MulticlassF1Score(num_classes=opt.num_classes, average='micro'),
                                'train_f1_macro': MulticlassF1Score(num_classes=opt.num_classes, average='macro'),
                                'train_precision': MulticlassPrecision(num_classes=opt.num_classes),
                                'train_recall': MulticlassRecall(num_classes=opt.num_classes),
                                }
                               ).to(opt.device)
    auroc = MulticlassAUROC(num_classes=opt.num_classes, average='macro').to(opt.device)

    start_time = timeit.default_timer()
    for i, (data, target, path) in enumerate(data_loader):
        data = data.to(opt.device)
        target = target.to(opt.device)
        optimizer.zero_grad()
        predictions = model(data)
        probs = F.softmax(predictions, dim=1)
        _, pred = torch.max(probs, dim=1)

        loss = F.nll_loss(probs, target)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        metrics(pred, target)
        global_target = torch.concatenate((global_target, target))
        global_pred = torch.concatenate((global_pred, pred))
        global_probs = torch.vstack((global_probs, probs))

        running_loss += loss.item() * data.size(0)

        if i % 5 == 0 and scheduler:
            wandb.log({"train_lr": scheduler.get_last_lr()[0]},
                      # commit=False, # Commit=False just accumulates data
                      )

    epoch_loss = running_loss / total_samples

    epoch_time = timeit.default_timer() - start_time
    print(f"Epoch time {epoch_time}")

    auroc(global_probs, global_target.long())
    global_target = global_target.cpu().detach().numpy()
    global_pred = global_pred.cpu().detach().numpy()
    global_probs = global_probs.cpu().detach().numpy()
    log_metrics = {
        **metrics.compute(),
        "train_epoch_loss": epoch_loss,
        "train_epoch_time": epoch_time,
        "train_confusion_matrix": wandb.plot.confusion_matrix(probs=global_probs, y_true=global_target,
                                                              class_names=['ellipse', 'square', 'triangle'],
                                                              title="Train confusion matrix"),
        "train_roc": wandb.plot.roc_curve(y_true=global_target, y_probas=global_probs,
                                          labels=['ellipse', 'square', 'triangle'],
                                          # TODO: Determine why classes_to_plot doesn't work with roc
                                          # classes_to_plot=['ellipse', 'square', 'triangle'],
                                          title="Train ROC", ),
        "train_auroc_macro": auroc.compute()
    }
    return log_metrics


def validation(model, data_loader, opt):
    model.eval()

    total_samples = len(data_loader.dataset)

    global_target = torch.tensor([], device=opt.device)
    global_pred = torch.tensor([], device=opt.device)
    global_probs = torch.empty((0, 3), device=opt.device)
    incorrect_img_paths = []
    incorrect_img_labels = torch.tensor([], device=opt.device)
    incorrect_img_predictions = torch.tensor([], device=opt.device)
    incorrect_images = torch.tensor([], device=opt.device)

    running_loss = 0.0

    metrics = MetricCollection({'val_f1_micro': MulticlassF1Score(num_classes=opt.num_classes, average='micro'),
                                'val_f1_macro': MulticlassF1Score(num_classes=opt.num_classes, average='macro'),
                                'val_precision': MulticlassPrecision(num_classes=opt.num_classes),
                                'val_recall': MulticlassRecall(num_classes=opt.num_classes),
                                }
                               ).to(opt.device)
    auroc = MulticlassAUROC(num_classes=opt.num_classes, average='macro').to(opt.device)
    start_time = timeit.default_timer()
    with torch.no_grad():
        for data, target, path in data_loader:
            data = data.to(opt.device)
            target = target.to(opt.device)
            res = model(data)
            probs = F.softmax(res, dim=1)
            probs = probs.to(opt.device)
            loss = F.nll_loss(probs, target, reduction='sum')
            _, pred = torch.max(probs, dim=1)

            incorrect_img_paths += [path[i] for i in range(len(path)) if pred[i] != target[i]]
            incorrect_img_labels = torch.concatenate((incorrect_img_labels, target[pred != target]))
            incorrect_img_predictions = torch.concatenate((incorrect_img_predictions, pred[pred != target]))
            incorrect_images = torch.concatenate((incorrect_images, data[pred != target]))

            metrics(pred, target)
            global_target = torch.concatenate((global_target, target))
            global_pred = torch.concatenate((global_pred, pred))
            global_probs = torch.vstack((global_probs, probs))

            running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / total_samples

    epoch_time = timeit.default_timer() - start_time

    auroc(global_probs, global_target.long())
    global_target = global_target.cpu().detach().numpy()
    global_pred = global_pred.cpu().detach().numpy()
    global_probs = global_probs.cpu().detach().numpy()
    incorrect_img_labels = incorrect_img_labels.cpu().detach().numpy()
    incorrect_img_predictions = incorrect_img_predictions.cpu().detach().numpy()
    incorrect_images = incorrect_images.cpu().detach().numpy()

    diff_mistakes = [int(re.search(r"(?<=diff)[0-9]", i).group()) for i in incorrect_img_paths]
    shapes_mistakes = [re.search(r"ellipse|triangle|square", i).group() for i in incorrect_img_paths]
    shape_diff_mistakes = [f"{s}_{d}" for s, d in zip(shapes_mistakes, diff_mistakes)]

    mistakes_data = [[incorrect_img_paths[i], diff_mistakes[i], shapes_mistakes[i],
                      wandb.Image(data_or_path=incorrect_images[i], caption=incorrect_img_paths[i]),
                      incorrect_img_predictions[i],
                      incorrect_img_labels[i]] for i in range(len(incorrect_img_paths))]
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
        "val_mistakes_by_diff_bar": wandb.plot.bar(
            table=wandb.Table(data=np.asarray([[d, diff_mistakes.count(d)] for d in range(1, 5)]),
                              columns=["difficulty", "mistakes"]),
            value="mistakes", label="difficulty", title="Mistakes by difficulty"),
        "val_mistakes_by_shape_bar": wandb.plot.bar(
            table=wandb.Table(data=np.asarray([[d, shapes_mistakes.count(d)] for d in set(shapes_mistakes)]),
                              columns=["shapes", "mistakes"]),
            value="mistakes", label="shapes", title="Mistakes by shape"),
        "val_mistakes_by_shape_diff_bar": wandb.plot.bar(
            table=wandb.Table(data=np.asarray([[d, shape_diff_mistakes.count(d)] for d in set(shape_diff_mistakes)]),
                              columns=["shape_and_difficulty", "mistakes"]),
            value="mistakes", label="shape_and_difficulty", title="Mistakes by shape and difficulty"),
        "val_mistakes_table": wandb.Table(data=mistakes_data,
                                          columns=['path', 'difficulty', 'shape', 'image', 'prediction', 'label']),
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
    parser.add_argument('-seed_dataset', '--seed_dataset', type=int, default=-1, help="Set random seed for dataset")

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
    parser.add_argument('-e', '--n_epochs', type=int, default=20, help="Max number of epochs for the current model")
    parser.add_argument('-max_e', '--max_epochs', type=int, default=20, help="Maximum number of epochs for all models")
    parser.add_argument('-min_e', '--min_epochs', type=int, default=5, help="Minimum number of epochs for all models")
    parser.add_argument('-nm', '--n_models', type=int, default=50, help="Number of models to be trained")
    parser.add_argument('-pp', '--parallel_processes', type=int, default=1,
                        help="Number of parallel processes to spawn for models [0 for all available cores]")
    parser.add_argument('-seed_everything', '--seed_everything', type=int, default=-1,
                        help="Set random seed for everything")
    parser.add_argument('-min_score', '--min_score', type=int, default=50,
                        help="Minimum score up to which the models will be trained")
    parser.add_argument('-max_score', '--max_score', type=int, default=99,
                        help="Maximum score up to which the models will be trained")
    parser.add_argument('-model_max_score', '--max_score', type=int, default=99,
                        help="Maximum score up to which the current model will be trained")
    parser.add_argument('-score_step', '--score_step', type=int, default=1,
                        help="Step between two nearest scores of consecutive models, up to which they are trained")

    # Optimizer options
    parser.add_argument('-optim', '--optimizer', type=str.lower, default="adamw",
                        choices=optimizer_choices.keys(),
                        help=f'Optimizer to be used {optimizer_choices.keys()}')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.05, help="Weight decay for optimizer")
    parser.add_argument('-momentum', '--momentum', type=float, default=0.9,
                        help="Momentum value for optimizers like SGD")

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
    epochs = [e for e in range(10, opt.max_epochs)]
    epoch_ranges = list(islice(cycle(epochs), opt.n_models))
    epoch_splits = [epoch_ranges[i:i + chunk] for i in range(0, len(epoch_ranges), chunk)]
    model_id_splits = [model_ids[i:i + chunk] for i in range(0, len(model_ids), chunk)]
    return epoch_splits, model_id_splits


# 2. Set the random seeds

def set_seed(seed_num):
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pass_right_constructor_arguments(target_class, opt):
    # TODO: Create an instance of a class by sending only the arguments that exist in the constructor
    pass


def create_experiments():
    parser = create_arg_parser()
    opt = parser.parse_args()

    model_ids = [f'model_{i}' for i in range(opt.n_models)]

    epoch_ranges = torch.linspace(opt.min_epochs, opt.max_epochs - 1, opt.n_models).long()
    # TODO: Add experiment description to args and log it in wandb

    functions_iter = repeat(run_experiment)
    args_iter = zip(model_ids)
    kwargs_iter = [
        {
            'max_score': ((i // 10) * 10 + ((i % 10) // opt.score_step) * opt.score_step) / 100,
            'max_epoch': opt.max_epochs
        } for i in torch.linspace(opt.min_score, opt.max_score, opt.n_models).long()
    ]

    if opt.parallel_processes <= 1:
        # It is faster to run the experiments on the main process if only one process should be used
        for f, args, kwargs in zip(functions_iter, args_iter, kwargs_iter):
            _proc_starter(f, args, kwargs)
    else:
        with mp.Pool(opt.parallel_processes) as pool:
            pool.starmap(_proc_starter, zip(functions_iter, args_iter, kwargs_iter))


def _proc_starter(f, args, kwargs):
    f(*args, **kwargs)


def run_experiment(model_id, max_epoch=100, max_score=1):
    # Model options
    model_choices = {CnnV1.__name__.lower(): CnnV1, }  # TODO: Add more model choices

    optimizer_choices = {optim.AdamW.__name__.lower(): optim.AdamW,
                         optim.SGD.__name__.lower(): optim.SGD}  # TODO: Add more optimizer choices

    # Scheduler options
    scheduler_choices = {
        optim.lr_scheduler.CyclicLR.__name__.lower(): optim.lr_scheduler.CyclicLR, }  # TODO: Add more scheduler choices

    parser = create_arg_parser(model_choices=model_choices, optimizer_choices=optimizer_choices,
                               scheduler_choices=scheduler_choices)
    opt = parser.parse_args()

    if opt.seed_everything >= 0:
        opt.seed_dataset = opt.seed_everything
        set_seed(opt.seed_everything)

    opt.n_epochs = max_epoch
    opt.model_max_score = max_score
    print(f'{model_id} is training with {max_score=} and {max_epoch=}')

    # Add specific options for experiments

    opt.device = 'cuda' if torch.cuda.is_available() and (opt.device == 'cuda') else 'cpu'
    print(opt.device)
    if opt.device == 'cuda':
        print(f'GPU {torch.cuda.get_device_name(0)}')

    train_loader, val_loader, test_loader = load_dataset(base_dir=opt.dataset, batch_size=opt.batch_size,
                                                         lengths=[opt.training_split, opt.validation_split,
                                                                  opt.evaluation_split],
                                                         shuffle=opt.shuffle, num_workers=opt.num_workers,
                                                         pin_memory=False,
                                                         seed=opt.seed_dataset
                                                         )

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

    # TODO: Optimize hyper-params with WandB Sweeper

    optimizer = optimizer_choices[opt.optimizer](params=model.parameters(), lr=opt.learning_rate,
                                                 weight_decay=opt.weight_decay)

    # TODO: Scheduler worked better when base and max values were reversed. We should look into that
    #   Maybe try base_lr of 0.001 with some other value for max_lr.
    scheduler = scheduler_choices[opt.scheduler](optimizer=optimizer, base_lr=opt.base_lr, max_lr=opt.max_lr,
                                                 step_size_up=opt.step_size_up,
                                                 cycle_momentum=opt.cycle_momentum, mode=opt.scheduler_mode)

    # TODO: Check model gradients. Make sure gradients are not vanishing/exploding
    #  wandb.watch(net, log='all')

    best_model_f1_macro = -np.Inf
    best_model_path = None
    artifact = wandb.Artifact(name=f'train-{opt.group}-{model_id}-max_epochs{opt.n_epochs}', type='model')

    # TODO: Add training resuming. This can be done from the model saved in wandb or from the local model
    for epoch in range(1, opt.n_epochs + 1):
        print(f"{epoch=}")
        train_metrics = train(model=model, optimizer=optimizer, data_loader=train_loader, opt=opt,
                              scheduler=scheduler)
        val_metrics = validation(model=model, data_loader=val_loader, opt=opt)

        # TODO: Add early stopping - Maybe not needed for this experiment. In that case log tables before ending
        last = epoch >= opt.n_epochs or val_metrics['val_f1_macro'] >= max_score

        if not last:
            del train_metrics["train_confusion_matrix"]
            del train_metrics["train_roc"]
            del val_metrics["val_confusion_matrix"]
            del val_metrics["val_roc"]
            del val_metrics["val_mistakes_by_diff_bar"]
            del val_metrics["val_mistakes_table"]
            del val_metrics["val_mistakes_by_shape_bar"]
            del val_metrics["val_mistakes_by_shape_diff_bar"]
        wandb.log(train_metrics)
        wandb.log(val_metrics)

        if val_metrics['val_f1_macro'] > best_model_f1_macro:
            print(f"Saving model with new best {val_metrics['val_f1_macro']=}")
            best_model_f1_macro, best_epoch = val_metrics['val_f1_macro'], epoch
            Path(f'experiments/{opt.group}').mkdir(exist_ok=True)
            new_best_path = os.path.join(f'experiments/{opt.group}',
                                         f'train-{opt.group}-{model_id}-max_epochs{opt.n_epochs}-epoch{epoch}'
                                         f'-max_metric{max_score}'
                                         f'-metric{val_metrics["val_f1_macro"]:.4f}.pt')
            torch.save(model.state_dict(), new_best_path)
            if best_model_path:
                os.remove(best_model_path)
            best_model_path = new_best_path

        if last:
            print(
                f"Finished training a {model_id=} with {max_epoch=} and {max_score=} "
                f"with va_f1_macro {val_metrics['val_f1_macro']}")
            break

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
    # TODO: Load model from wandb for the current run
    model.load_state_dict(torch.load(best_model_path))
    model.to(opt.device)

    # TODO: Create a new helper function that returns the number of model parameters
    # TODO: Also save number of parameters for the model in the wandb config
    # TODO: Save a model architecture that was used. This should include the layer information,
    #  similarly to how torch returns the architecture.
    #  Maybe even as an image if it can be visualized with some library
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print(pytorch_total_params)
    eval_metrics = validation(model=model, data_loader=test_loader, opt=opt)
    wandb.log(eval_metrics)
    wb_run_eval.finish()


def main():
    create_experiments()


if __name__ == "__main__":
    main()
