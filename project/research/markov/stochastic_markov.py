# Original file is from this repo https://github.com/milansegedinac/kst/blob/master/learning_spaces/kst/stochastic_markov.py
# It was created by nikola-difine

import operator
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _array2dict_vals(array: np.ndarray, dict: dict):
    for i, key in enumerate(dict):
        dict[key] = array[i]


def _scale_probabilites(states: dict[Tuple[str], float]):
    probabilites = np.array(list(states.values()))
    probabilites /= sum(probabilites)
    _array2dict_vals(probabilites, states)


def _likeliest_state(states: dict[Tuple[str], float]) -> Tuple[Tuple[str], float]:
    """
    Returns likeliest state and its probability.
    :return: (state, probability)
    """
    return max(states.items(), key=operator.itemgetter(1))


def _take_answer(question: str, model=None, data_path: str = None, opt=None) -> bool:
    print(f'{question}: correct/incorrect? [1/0]')
    # TODO: Get the question image and see if model can predict it correctly
    #  Example: Let the question be triangle_diff1.
    #  First randomly select an image that has the same shape and difficulty.
    #  After that, use the model to try to predict what shape is on that image.
    #  Return true or false based on the prediction

    # data, target, path = load_image_by_shape_and_difficulty(question, data_path)
    # data = data.to(opt.device)
    # target = target.to(opt.device)
    # res = model(data)
    # probs = F.softmax(res, dim=1)
    # probs = probs.to(opt.device)
    # _, pred = torch.max(probs, dim=1)
    # return pred == target
    return int(input()) == 1


def questioning_rule(states: dict[Tuple[str], float]) -> str:
    """
    :param states: dictionary mapping states (sets of problems/questions) to probabilities
    :return: question to be asked
    """
    if not np.isclose(1, sum(states.values()), atol=0.01):
        raise ValueError('Probabilities do not add up to 1!')

    state, _ = _likeliest_state(states)
    return random.choice(state)


def response_rule(question: str, states: dict[Tuple[str], float]) -> float:
    """
    :param question: question the answer is given to
    :param states: dictionary mapping states (sets of problems/questions) to probabilities
    :return: probability of giving correct answer according to given states
    """
    ret_val = 0
    for state, probability in states.items():
        if question in state:
            ret_val += probability
    return ret_val


def updating_rule(question: str, answer_correct: bool, r: float, states: dict[Tuple[str], float]):
    """
    Updates probabilites on passed states.
    :param question: question the answer is given to
    :param answer_correct: whether answer is correct
    :param r: response rule output
    :param states: dictionary mapping states (sets of problems/questions) to probabilities
    """
    theta = 0.1 * r
    theta_compl = 1 - theta
    if not answer_correct:
        theta, theta_compl = theta_compl, theta

    for state in states:
        if question in state:
            states[state] *= theta_compl
        else:
            states[state] *= theta
    _scale_probabilites(states)


def final_state(states: dict[Tuple[str], float]):
    state, probability = _likeliest_state(states)
    return state if probability > 0.75 else None


def stochastic_markov(states: dict[Tuple[str] | Tuple[str, str] | Tuple[str, str, str], float],
                      model=None, data_path=None, opt=None) -> Tuple[str]:
    max_iter = 100
    for _ in range(max_iter):
        question = questioning_rule(states)
        r = response_rule(question, states)
        answer_correct = _take_answer(question, model, data_path, opt)
        updating_rule(question, answer_correct, r, states)
        print(states)
        final = final_state(states)
        if final is not None:
            print(final)
            return
    print('Non-conclusive.')


def demo():
    states = {('a'): 0.125, ('a', 'b'): 0.25, ('b'): 0.125, ('a', 'b', 'c'): 0.5}
    print(states)
    stochastic_markov(states)


if __name__ == '__main__':
    demo()
