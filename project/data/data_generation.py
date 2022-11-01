
# TODO: Inert functions for data generation.
# TODO: Add a function that can be called to genrate data. Make it parametrized

import numpy as np
import cv2
import random
from datetime import datetime
import os


def generate_ellipse():
    center = np.array([IMG_DIM // 2, IMG_DIM // 2])
    MAIN_AXIS_LENGHT_THRESHOLD = 25
    SIDE_AXIS_LENGHT_THRESHOLD = 25
    TRANSLATE_RANGE = IMG_DIM // 4
    while True:
        # TODO treba generisati kontrast elipse?
        # TODO napraviti da glavna i sporedna osa ne mogu da odstupaju previše jedna od druge?
        # TODO ubaciti formulu za generisanje svih tacki na elipsi
        ang = angle()  # returns angle between 0 and 360
        startAngle = 0
        endAngle = 360  # creates pacman if less than 360

        h_axis, v_axis = random.randint(MAIN_AXIS_LENGHT_THRESHOLD, IMG_DIM // 4), random.randint(
            MAIN_AXIS_LENGHT_THRESHOLD, IMG_DIM // 4)  # axis length

        transl_h, transl_v = random.randint(-TRANSLATE_RANGE, TRANSLATE_RANGE), random.randint(-TRANSLATE_RANGE,
                                                                                               TRANSLATE_RANGE)
        center = center + np.array([transl_h, transl_v])  # Translate the center

        h_left_pt, h_right_pt = center - np.array([h_axis, 0]), center + np.array([h_axis, 0])
        v_bot_pt, v_top_pt = center - np.array([0, v_axis]), center + np.array([0, v_axis])

        points = np.vstack((h_left_pt, h_right_pt, v_bot_pt, v_top_pt))

        points = rotate(points, ang)  # Returns rotated end-of-axis points
        if np.any(points < 0) or np.any(points > IMG_DIM):  # TODO check if this works as intended
            continue

        print(points)
        # Color in BGR
        color = 120  # TODO: Determine how to select color for shapes based on background and other shape colors if they exist on the image

        # Line thickness of -1 px
        thickness = -1

        return center, h_axis, v_axis, ang, startAngle, endAngle, color, thickness
