import numpy as np
import cv2
import random
from datetime import datetime
import os
import matplotlib.pyplot as plt


def pt():
    return random.randint(0, 30)


def line_pt():
    return random.randint(0, 350)


def angle():
    return random.randint(0, 360)


def rotate(points, angle):
    ANGLE = np.deg2rad(angle)
    c_x, c_y = np.mean(points, axis=0)
    return np.array(
        [
            [
                c_x + np.cos(ANGLE) * (px - c_x) - np.sin(ANGLE) * (py - c_x),
                c_y + np.sin(ANGLE) * (px - c_y) + np.cos(ANGLE) * (py - c_y)
            ]
            for px, py in points
        ]
    ).astype(int)


def get_translate_range(_points, img_dim):
    max_offsets = [img_dim, img_dim] - np.max(_points, axis=0)
    min_offsets = -np.min(_points, axis=0)
    offsets = [(min_offsets[0], max_offsets[0]),
               (min_offsets[1], max_offsets[1])]
    return offsets


def translate(points, img_dim):
    offsets = get_translate_range(points, img_dim)
    x = points + [random.randint(*offsets[0]), random.randint(*offsets[1])]
    return x


def scale(points, scaling_factor=1):
    scale_matrix = np.eye(2) * scaling_factor
    return np.array([scale_matrix @ point for point in points]).astype(int)


""" 
  This function generates random noise over an image
"""


def noisy(image, noise_type='gauss'):
    if noise_type == "gauss":
        row, col = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss

        return noisy

    elif noise_type == "s&p":
        row, col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0

        return out

    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)

        return noisy

    elif noise_type == "speckle":
        img = image.copy()
        row, col = image.shape
        gauss = np.random.randn(row, col)
        gauss = gauss.reshape(row, col)
        noisy = image + img * gauss
        return noisy
