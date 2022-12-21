import os
from itertools import repeat
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import optim

import wandb
from data.data_loader import load_dataset_curriculum
from models.cnn_v1 import CnnV1
from train import train, validation, set_seed, create_arg_parser, create_wandb_val_plots, \
    del_wandb_val_untracked_metrics, del_wandb_train_untracked_metrics, create_wandb_train_plots


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
            'max_score': opt.max_score,
            'max_epoch': opt.max_epochs
        } for i in torch.linspace(opt.min_score, opt.max_score, opt.n_models).long()
    ]

    if opt.parallel_processes <= 1:
        # It is faster to run the experiments on the main process if only one process should be used
        for f, args, kwargs in zip(functions_iter, args_iter, kwargs_iter):
            _proc_starter(f, args, kwargs)
    else:
        failed_process_args_kwargs = []
        with mp.Pool(opt.parallel_processes) as pool:
            for f, ret_args, ret_kwargs, process_failed in pool.starmap(_proc_starter,
                                                                        zip(functions_iter, args_iter, kwargs_iter)):
                if process_failed:
                    failed_process_args_kwargs.append((f, ret_args, ret_kwargs))
        print(f"Failed models: {len(failed_process_args_kwargs)}.")
        for f in failed_process_args_kwargs:
            print(f"Failed model: {f[1][0]}")
        n_retry_attempts = opt.n_models
        while failed_process_args_kwargs and n_retry_attempts > 0:
            val = failed_process_args_kwargs.pop(0)
            f, ret_args, ret_kwargs, process_failed = _proc_starter(val[0], val[1], val[2])
            if process_failed:
                failed_process_args_kwargs.append((f, ret_args, ret_kwargs))
            n_retry_attempts -= 1


def _proc_starter(f, args, kwargs):
    return f, *f(*args, **kwargs)


def debug_dataloader_samplers(data_loader, hierarchy_level, knowledge_hierarchy):
    count = 0
    for data, target, paths in data_loader:
        for path in paths:
            name = Path(path).stem.split("_")
            key = name[0] + "_" + name[-1]
            if knowledge_hierarchy[key] == hierarchy_level:
                count += 1
    return count / len(data_loader.dataset)


def run_experiment(model_id, max_epoch=100, max_score=1, *args, **kwargs):
    # Model options
    model_choices = {CnnV1.__name__.lower(): CnnV1, }  # TODO: Add more model choices

    optimizer_choices = {optim.AdamW.__name__.lower(): optim.AdamW,
                         optim.SGD.__name__.lower(): optim.SGD}  # TODO: Add more optimizer choices

    # Scheduler options
    scheduler_choices = {
        optim.lr_scheduler.CyclicLR.__name__.lower(): optim.lr_scheduler.CyclicLR, }  # TODO: Add more scheduler choices

    parser = create_arg_parser(model_choices=model_choices, optimizer_choices=optimizer_choices,
                               scheduler_choices=scheduler_choices)
    # Parser option specific for this experiment
    parser.add_argument('-initial_p', '--initial_p', type=float, default=0.7,
                        help="Initial probability for geometric distribution for experiment with "
                             "curriculum learning that uses weighted random sampler")
    parser.add_argument('-epoch_per_curriculum', '--epoch_per_curriculum', type=int, default=4,
                        help="How many epochs to train before switching to a new curriculum")
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

    p = [opt.initial_p * (1 - opt.initial_p) ** i for i in range(6)]
    knowledge_hierarchy = {
        "triangle_diff1": 0,
        "triangle_diff2": 2,
        "triangle_diff3": 5,
        "triangle_diff4": 5,
        "square_diff1": 0,
        "square_diff2": 1,
        "square_diff3": 2,
        "square_diff4": 2,
        "ellipse_diff1": 1,
        "ellipse_diff2": 2,
        "ellipse_diff3": 3,
        "ellipse_diff4": 4,
    }

    train_loader, val_loader, test_loader = load_dataset_curriculum(base_dir=opt.dataset, batch_size=opt.batch_size,
                                                                    lengths=[opt.training_split, opt.validation_split,
                                                                             opt.evaluation_split],
                                                                    shuffle=opt.shuffle, num_workers=opt.num_workers,
                                                                    pin_memory=False,
                                                                    seed=opt.seed_dataset,
                                                                    p=p,
                                                                    knowledge_hierarchy=knowledge_hierarchy,
                                                                    )

    # print(debug_dataloader_samplers(val_loader, 2, knowledge_hierarchy))

    # TODO: Determine optimal step_size_up for cyclicLR scheduler.
    if opt.step_size_up <= 0:
        opt.step_size_up = 2 * len(train_loader.dataset) // opt.batch_size

    wb_run_train = wandb.init(entity=opt.entity, project=opt.project_name, group=opt.group,
                              # save_code=True, # Pycharm complains about duplicate code fragments
                              job_type=opt.job_type,
                              # TODO: Add tags as arguments for argparser
                              tags=['variable_max_score'],
                              name=f'{model_id}_train_max_score_{round(float(max_score), 2)}',
                              config=opt,
                              )

    # Define model
    model = model_choices[opt.model](depth=opt.depth, in_channels=opt.in_channels, out_channels=opt.out_channels,
                                     kernel_dim=opt.kernel_dim, mlp_dim=opt.mlp_dim, padding=opt.padding,
                                     stride=opt.stride, max_pool=opt.max_pool,
                                     dropout=opt.dropout)

    model = model.to(opt.device)

    # TODO: Optimize hyper-params with WandB Sweeper
    # TODO: Check what params were used for AdamW in the paper that achieved similar performance to SGD with momentum
    #  on ImageNet. Also check if they used LSScheduler with AdamW.

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
    artifact = wandb.Artifact(
        name=f'{model_id}.pt',
        type='model')
    p_values = [opt.initial_p / 2 ** i for i in range(1, 6)]
    try:
        # TODO: Add training resuming. This can be done from the model saved in wandb or from the local model
        for epoch in range(1, opt.n_epochs + 1):
            print(f"{epoch=}")

            # TODO: Change dataloader to use different weights based on passed epochs
            if epoch % opt.epoch_per_curriculum == 0:
                selected_p = p_values[min(epoch // 3 - 1, len(p_values) - 1)]
                p = [selected_p * (1 - selected_p) ** i for i in range(6)]
                train_loader, val_loader, test_loader = load_dataset_curriculum(base_dir=opt.dataset,
                                                                                batch_size=opt.batch_size,
                                                                                lengths=[opt.training_split,
                                                                                         opt.validation_split,
                                                                                         opt.evaluation_split],
                                                                                shuffle=opt.shuffle,
                                                                                num_workers=opt.num_workers,
                                                                                pin_memory=False,
                                                                                seed=opt.seed_dataset,
                                                                                p=p,
                                                                                knowledge_hierarchy=knowledge_hierarchy,
                                                                                )
            train_metrics = train(model=model, optimizer=optimizer, data_loader=train_loader, opt=opt,
                                  scheduler=scheduler)
            val_metrics = validation(model=model, data_loader=val_loader, opt=opt, save_images=opt.save_val_images)

            # TODO: Add early stopping - Maybe not needed for this experiment. In that case log tables before ending
            last = epoch >= opt.n_epochs or val_metrics['val_f1_macro'] >= max_score

            if last:
                train_metrics.update(create_wandb_train_plots(train_metrics=train_metrics))
                val_metrics.update(create_wandb_val_plots(val_metrics=val_metrics, save_images=opt.save_val_images))

            del_wandb_train_untracked_metrics(train_metrics=train_metrics)
            del_wandb_val_untracked_metrics(val_metrics=val_metrics)

            wandb.log(train_metrics)
            wandb.log(val_metrics)

            if val_metrics['val_f1_macro'] > best_model_f1_macro:
                print(f"Saving model with new best {val_metrics['val_f1_macro']=}")
                best_model_f1_macro, best_epoch = val_metrics['val_f1_macro'], epoch
                Path(f'../../experiments/{opt.group}').mkdir(exist_ok=True)
                new_best_path = os.path.join(f'../../experiments/{opt.group}',
                                             f'train-{opt.group}-{model_id}-max_epochs{opt.n_epochs}-epoch{epoch}'
                                             f'-max_metric{round(float(max_score), 2)}'
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

    except FileNotFoundError as e:
        wb_run_train.finish()
        # wb_run_train.delete()  # Delete train run if an error has occurred
        print(f"Exception happened for model {model_id}\n {e}")
        return [model_id, *args], {"max_epoch": max_epoch, "max_score": max_score,
                                   **kwargs}, True  # Run Failed is True

    # Test loading
    opt.job_type = "eval"
    wb_run_eval = wandb.init(entity=opt.entity, project=opt.project_name, group=opt.group,
                             # save_code=True, # Pycharm complains about duplicate code fragments
                             job_type=opt.job_type,
                             # TODO: Replace with tags argument from argparser once its added
                             tags=['variable_max_score'],
                             name=f'{model_id}_eval_max_score_{round(float(max_score), 2)}',
                             config=opt,
                             )

    model = model_choices[opt.model](depth=opt.depth, in_channels=opt.in_channels, out_channels=opt.out_channels,
                                     kernel_dim=opt.kernel_dim, mlp_dim=opt.mlp_dim, padding=opt.padding,
                                     stride=opt.stride, max_pool=opt.max_pool,
                                     dropout=opt.dropout)
    # TODO: Load model from wandb for the current run
    model.load_state_dict(torch.load(best_model_path))
    model.to(opt.device)
    try:
        # TODO: Create a new helper function that returns the number of model parameters
        # TODO: Also save number of parameters for the model in the wandb config
        # TODO: Save a model architecture that was used. This should include the layer information,
        #  similarly to how torch returns the architecture.
        #  Maybe even as an image if it can be visualized with some library
        # pytorch_total_params = sum(p.numel() for p in model.parameters())
        # print(pytorch_total_params)
        eval_metrics = validation(model=model, data_loader=test_loader, opt=opt, save_images=opt.save_test_images)
        eval_metrics.update(create_wandb_val_plots(val_metrics=eval_metrics, save_images=opt.save_test_images))
        del_wandb_val_untracked_metrics(val_metrics=eval_metrics)
        wandb.log(eval_metrics)
        wb_run_eval.finish()
    except FileNotFoundError as e:
        wb_run_eval.finish()
        # wb_run_eval.delete()  # Delete eval run if an error has occurred
        # wb_run_train.delete()  # Delete train run also if an error has occurred
        print(f"Exception happened for model {model_id}\n {e}")
        return [model_id, *args], {"max_epoch": max_epoch, "max_score": max_score,
                                   **kwargs}, True  # Run Failed is True
    # TODO: Add our own code for removing models from the wandb folder during training.
    return [model_id, *args], {"max_epoch": max_epoch, "max_score": max_score, **kwargs}, False  # Run Failed is False


def main():
    create_experiments()


if __name__ == "__main__":
    main()
