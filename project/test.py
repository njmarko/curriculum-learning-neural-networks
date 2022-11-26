# TODO: Add test loop
# TODO: Log experiment results with wandb
# TODO: Save models and results in experiments folder
# TODO: Create argparser for all parameters that can be defined

import json
import torch

from collections import defaultdict
from data.data_loader import load_dataset_iita
import pandas as pd
from learning_spaces.kst import iita

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_iita_column():
    return torch.zeros((500, ), dtype=torch.int32)

def load_models():
    pass

def get_problem_category(path):
    pass

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
    models = load_models()
    dataset = load_dataset_iita()

    return get_knowledge_table(models, dataset)


def run_iita():
    knowledge_table = prepare_iita()
    knowledge_table_df = pd.DataFrame(knowledge_table)

    response = iita(knowledge_table_df, v=3)
    response_json = json.dumps(response, indent=4)
 
    with open("iita_results.json", "w") as outfile:
        outfile.write(response_json)
        print(response)