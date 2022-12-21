import torch
from torch import optim

from models.cnn_v1 import CnnV1
from research.markov.stochastic_markov import stochastic_markov
from train import create_arg_parser


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

    # TODO: How to choose initial state probabilities?
    states = {'triangle_diff1': 0.125, ('triangle_diff1', 'square_diff1'): 0.25, 'square_diff1': 0.125,
              ('triangle_diff1', 'square_diff1', 'ellipse_diff1'): 0.5}
    data_path = opt.dataset
    stochastic_markov(states, model, data_path)


if __name__ == "__main__":
    main()
