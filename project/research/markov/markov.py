from itertools import chain, combinations

import torch
from torch import optim

from models.cnn_v1 import CnnV1
from research.markov.stochastic_markov import stochastic_markov, load_markov_dataset
from train import create_arg_parser


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def main():
    model_choices = {CnnV1.__name__.lower(): CnnV1, }  # TODO: Add more model choices

    optimizer_choices = {optim.AdamW.__name__.lower(): optim.AdamW,
                         optim.SGD.__name__.lower(): optim.SGD}  # TODO: Add more optimizer choices

    # Scheduler options
    scheduler_choices = {
        optim.lr_scheduler.CyclicLR.__name__.lower(): optim.lr_scheduler.CyclicLR, }  # TODO: Add more scheduler choices

    parser = create_arg_parser(model_choices=model_choices, optimizer_choices=optimizer_choices,
                               scheduler_choices=scheduler_choices)

    parser.add_argument('-used_model_path', '--used_model_path', type=str,
                        default="../../experiments/variable_max_score_v3/"
                                "train-variable_max_score_v3-model_498-max_epochs20-epoch20-max_metric0.98-metric0.9320"
                                ".pt",
                        help="Path to the model that will be used in this experiment")

    opt = parser.parse_args()

    model = model_choices[opt.model](depth=opt.depth, in_channels=opt.in_channels, out_channels=opt.out_channels,
                                     kernel_dim=opt.kernel_dim, mlp_dim=opt.mlp_dim, padding=opt.padding,
                                     stride=opt.stride, max_pool=opt.max_pool,
                                     dropout=opt.dropout)

    model.load_state_dict(torch.load(opt.used_model_path))
    model.to(opt.device)
    model.eval()

    states = {
        ('ellipse_1'): 1 / 12,
        ('ellipse_2'): 1 / 12,
        ('ellipse_3'): 1 / 12,
        ('ellipse_4'): 1 / 12,
        ('square_1'): 1 / 12,
        ('square_2'): 1 / 12,
        ('square_3'): 1 / 12,
        ('square_4'): 1 / 12,
        ('triangle_1'): 1 / 12,
        ('triangle_2'): 1 / 12,
        ('triangle_3'): 1 / 12,
        ('triangle_4'): 1 / 12,
    }

    dataset = load_markov_dataset('../../' + opt.dataset)

    powset = list(powerset(states.keys()))[1:]
    states = dict(zip(powset, [1 / len(powset)] * len(powset)))
    stochastic_markov(states, model, dataset, opt)


if __name__ == "__main__":
    main()
