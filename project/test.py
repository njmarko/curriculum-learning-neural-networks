# TODO: Add test loop
# TODO: Log experiment results with wandb
# TODO: Save models and results in experiments folder
# TODO: Create argparser for all parameters that can be defined

import os
import json
import torch
import argparse

from collections import defaultdict
from data.data_loader import load_dataset_iita
import pandas as pd
from learning_spaces.kst import iita
from models.cnn_v1 import CnnV1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

N_MODELS = 500
MODEL_PATH = 'project/experiments/variable_max_score_v3'

def create_arg_parser(model_choices=None):
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
    parser.add_argument('-d', '--dataset', type=str, default="generated_images/iita_dataset",
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
    parser.add_argument('-model_max_score', '--model_max_score', type=int, default=99,
                        help="Maximum score up to which the current model will be trained")
    parser.add_argument('-score_step', '--score_step', type=int, default=1,
                        help="Step between two nearest scores of consecutive models, up to which they are trained")
    parser.add_argument('-save_val_images', '--save_val_images', type=bool, default=False,
                        help="Save validation images on which the model made mistakes in the last epoch")
    parser.add_argument('-save_test_images', '--save_test_images', type=bool, default=False,
                        help="Save test images on which the model made mistakes.")

    return parser

# TODO: Izmeniti N_MODELS u opt.n_models
def get_iita_column():
    return torch.zeros((N_MODELS, ), dtype=torch.int32)

def load_models(opt):

    paths = [os.path.join(MODEL_PATH, fn) for fn in next(os.walk(MODEL_PATH))[2]]
    models = []
    # cnt = 0
    for path in paths:
        # if cnt == 20: break
        # cnt += 1
        print(path)

        model = CnnV1(depth=opt.depth, in_channels=opt.in_channels, out_channels=opt.out_channels,
                                        kernel_dim=opt.kernel_dim, mlp_dim=opt.mlp_dim, padding=opt.padding,
                                        stride=opt.stride, max_pool=opt.max_pool,
                                        dropout=opt.dropout)

        # TODO: Izmeniti na model.to(opt.device) i izbrisati drugi parametar iz prve linije (map_location)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.to(device)
        models.append(model)

    return models

def get_problem_category(path):
    path = path[0]
    filename = (path.split('/')[-1]).split('.')[0]
    shape, _, diff = filename.split('_')
    return f'{shape}_{diff}'

#TODO: Izmeniti .to(device) u .to(opt.device)
def get_knowledge_table(models, dataset):
    scores_dict = defaultdict(get_iita_column)

    for data, label, path in dataset:
        data = data.to(device)
        label = label.to(device)
        category = get_problem_category(path)

        for idx, model in enumerate(models):
            scores = model(data)
            _, prediction = scores.max(dim=1)
            scores_dict[category][idx] = int(prediction == label)

    return scores_dict 


def prepare_iita():
    opt = create_arg_parser().parse_args()

    dataset = load_dataset_iita(opt.dataset)
    models = load_models(opt)

    return get_knowledge_table(models, dataset)


def run_iita():
    knowledge_table = prepare_iita()
    knowledge_table_df = pd.DataFrame(knowledge_table)
    column_names = knowledge_table_df.columns.values.tolist()
    print(column_names)

    response = iita(knowledge_table_df, v=3)
    print(response)
    with open('iita_results.txt', 'w') as outfile:
        outfile.write(response.__repr__())
    with open('iita_results_graph.txt', 'w') as outfile:
        outfile.write(response['implications'].__repr__())
    with open('formatted_iita_graph.txt', 'w') as outfile:
        for pair in response['implications']:
            outfile.write(f'{column_names[pair[0]]} {column_names[pair[1]]}\n')
    # response_json = json.dumps(response, indent=4)
 
    # with open("iita_results.json", "w") as outfile:
    #     outfile.write(response_json)
    #     print(response)

run_iita()