# TODO: Inert functions for data generation.
# TODO: Add a function that can be called to genrate data. Make it parametrized
# TODO: Add colors for shapes
# TODO: Parametrize with ArgumentParser

import os
import random
import time
from itertools import cycle, islice, product
from pathlib import Path

import cv2
import numpy as np
from generation_utils import (angle, get_translate_range, noisy,
                              rotate, scale, translate)
from data_loader import load_dataset

IMG_DIM = 96
MAX_OFFSET = 10 # TODO: Not used. Remove if not needed
DATASET_SIZE = 1000
DATASET_BASE_DIR = 'generated_images/dataset5/'
COLOR_GRADIENT_THRESHOLD = 50

def get_difficulty_level():
    print('Difficulty level:')
    print('\t- 1 - Regular shapes')
    print('\t- 2 - Irregular shapes')
    print('\t- 3 - Irregular with occlusion')
    print('\t- 4 - Irregular with occlusion and noise')
    print()
    print('Input difficulty level:')
    level = int(input('>>> '))
    print(f'You have chosen level {level}')
    return level


def get_shape():
    print('Shape:')
    print('\t- 1 - Triangle')
    print('\t- 2 - Quadrilateral')
    print('\t- 3 - Ellipse')
    print()
    print('Input shape:')
    shape_no = int(input('>>> '))

    if shape_no == 1:
        shape = 'triangle'
    elif shape_no == 2:
        shape = 'quad'
    else:
        shape = 'ellipse'

    print(f'You have chosen shape {shape}')
    return shape


""" 
  This function generates points for an equilateral triangle

"""


def get_triangle_points():
    top_point_x = IMG_DIM // 2
    top_point_y = IMG_DIM // 4
    bottom_point_y = top_point_y + IMG_DIM // 3

    triangle_height = IMG_DIM // 3
    triangle_side = 2 * triangle_height * np.sqrt(3) / 3
    left_bottom_point_x = int(top_point_x - triangle_side / 2)
    right_bottom_point_x = int(top_point_x + triangle_side / 2)

    return np.array([
        (top_point_x, top_point_y),
        (left_bottom_point_x, bottom_point_y),
        (right_bottom_point_x, bottom_point_y),
    ])


""" 
  This function calculates the lengths of sides of a triangle

"""


def calculate_sides(p1, p2, p3):
    x_dist_1 = (p1[0] - p2[0]) ** 2
    x_dist_2 = (p1[0] - p3[0]) ** 2
    x_dist_3 = (p2[0] - p3[0]) ** 2

    y_dist_1 = (p1[1] - p2[1]) ** 2
    y_dist_2 = (p1[1] - p3[1]) ** 2
    y_dist_3 = (p2[1] - p3[1]) ** 2

    return (
        np.sqrt(x_dist_1 + y_dist_1),
        np.sqrt(x_dist_2 + y_dist_2),
        np.sqrt(x_dist_3 + y_dist_3),
    )


"""
  This function calculates the angle between two sides specified by 's1' and 's2',
  that is, opposite to side specified by 'opposite'
  The length is calculated using cosine theorem
"""


def calculate_angle(opposite, s1, s2):
    cos_gamma = (s1 ** 2 + s2 ** 2 - opposite ** 2) / (2 * s1 * s2)
    return np.rad2deg(np.arccos(cos_gamma))


"""
  This function generates a triangle
  It randomly moves the points of an equilateral triangle
  It checks wether the new points are within the bounds of the image,
  wether they are reasonably long and wether the angles are reasonably sharp

"""


def generate_triangle(difficulty=1):
    SIDE_THRESHOLD = 5
    ANGLE_TRESHOLD = 150
    TRANSLATE_RANGE = IMG_DIM
    SCALE_RANGE = 2
    points = get_triangle_points()

    while True:
        if difficulty == 1:
            new_points = translate(points.copy(), IMG_DIM)
            new_points = rotate(new_points, random.randint(0, 180))
            new_points = scale(new_points, random.uniform(0.5, SCALE_RANGE))
        else:
            p1_translate = get_translate_range([points[0]], IMG_DIM)
            p2_translate = get_translate_range([points[1]], IMG_DIM)
            p3_translate = get_translate_range([points[2]], IMG_DIM)
            mv = [
                [random.randint(*p1_translate[0]), random.randint(*p1_translate[1])],
                [random.randint(*p2_translate[0]), random.randint(*p2_translate[1])],
                [random.randint(*p3_translate[0]), random.randint(*p3_translate[1])],
            ]

            new_points = points + mv

        sides = calculate_sides(new_points[0], new_points[1], new_points[2])

        angle1 = calculate_angle(sides[0], sides[1], sides[2])
        angle2 = calculate_angle(sides[1], sides[0], sides[2])
        angle3 = calculate_angle(sides[2], sides[0], sides[1])

        if np.any(new_points < 0) or np.any(new_points > IMG_DIM):
            continue

        if np.any(np.array(sides) <= SIDE_THRESHOLD):
            continue

        if angle1 >= ANGLE_TRESHOLD or angle2 >= ANGLE_TRESHOLD or angle3 >= ANGLE_TRESHOLD:
            continue

        return new_points


"""
  This function generates a square

"""


def get_square_points():
    left_x = IMG_DIM // 4
    right_x = 3 * (IMG_DIM // 4)
    top_y = IMG_DIM // 4
    bottom_y = 3 * (IMG_DIM // 4)

    return np.array([
        (left_x, top_y),
        (right_x, top_y),
        (right_x, bottom_y),
        (left_x, bottom_y),
    ])


"""
  This function generates a rando parallelogram
  It checks wether the sides are reasonably long and withing bounds
  It randomly rotates the generated parallelogram, and checks wether the
  rotated quadrilateral is within image bounds
"""


def generate_square(difficulty=2):
    points = get_square_points()
    TRANSLATE_RANGE = IMG_DIM // 2
    HEIGHT_THRESHOLD = 5
    SIDE_THRESHOLD = 5

    while True:
        if difficulty == 1:
            new_points = translate(points, IMG_DIM)
            new_points = scale(new_points, random.uniform(0.5, 2))
        else:
            translate_top_left_x = random.randint(-TRANSLATE_RANGE, TRANSLATE_RANGE)
            translate_top_right_x = random.randint(-TRANSLATE_RANGE, TRANSLATE_RANGE)
            translate_bottom_left_x = random.randint(-TRANSLATE_RANGE, TRANSLATE_RANGE)
            translate_bottom_right_x = random.randint(-TRANSLATE_RANGE, TRANSLATE_RANGE)
            translate_top_y = random.randint(-TRANSLATE_RANGE, TRANSLATE_RANGE)
            translate_bottom_y = random.randint(-TRANSLATE_RANGE, TRANSLATE_RANGE)

            mv = [
                [translate_top_left_x, translate_top_y],
                [translate_top_right_x, translate_top_y],
                [translate_bottom_right_x, translate_bottom_y],
                [translate_bottom_left_x, translate_bottom_y]
            ]

            new_points = points + mv

            side_lenghts = np.array([
                new_points[0][0] - new_points[1][0],
                new_points[3][0] - new_points[2][0]
            ])

            if np.any(side_lenghts <= SIDE_THRESHOLD):
                continue
            if np.abs(new_points[0][1] - new_points[2][1] < HEIGHT_THRESHOLD):
                continue

        new_points = rotate(new_points, angle())
        if np.any(new_points < 0) or np.any(new_points > IMG_DIM):
            continue
        return new_points


"""
  This function generates a line that crosses a shape
"""


def get_overlapping_line(points):
    # starting_point = [random.randint(0, IMG_DIM), random.randint(0, IMG_DIM)]
    # ending_point = [random.randint(0, IMG_DIM), random.randint(0, IMG_DIM)]
    # return (
    #     starting_point,
    #     ending_point
    # )

    while True:
        top_border = np.min(points, axis=0)[1]
        bottom_border = np.max(points, axis=0)[1]
        left_border = np.min(points, axis=0)[0]
        right_border = np.max(points, axis=0)[0]

        starting_point = [random.randint(0, left_border), random.randint(0, IMG_DIM)]
        ending_point = [random.randint(right_border, IMG_DIM), random.randint(0, IMG_DIM)]

        return (
            starting_point,
            ending_point
        )


def generate_ellipse(difficulty):
    center_original = np.array([IMG_DIM // 2, IMG_DIM // 2])
    MAIN_AXIS_LENGHT_THRESHOLD = 5
    SIDE_AXIS_LENGHT_THRESHOLD = 5
    TRANSLATE_RANGE = IMG_DIM // 4
    total_errors = 0
    total_time_in_errors = 0
    while True:
        # TODO treba generisati kontrast elipse?
        # TODO napraviti da glavna i sporedna osa ne mogu da odstupaju previše jedna od druge?
        # TODO ubaciti formulu za generisanje svih tacki na elipsi
        start = time.time()
        center = center_original.copy()
        ang = angle()  # returns angle between 0 and 360
        startAngle = 0
        endAngle = 360  # creates pacman if less than 360

        if difficulty == 1:
            h_axis = v_axis = random.randint(MAIN_AXIS_LENGHT_THRESHOLD, IMG_DIM // 4)
        else:
            h_axis, v_axis = random.randint(MAIN_AXIS_LENGHT_THRESHOLD, IMG_DIM // 4), random.randint(
                SIDE_AXIS_LENGHT_THRESHOLD, IMG_DIM // 4)  # axis length

        transl_h, transl_v = random.randint(-TRANSLATE_RANGE, TRANSLATE_RANGE), random.randint(-TRANSLATE_RANGE,
                                                                                               TRANSLATE_RANGE)
        center = center + np.array([transl_h, transl_v])  # Translate the center

        h_left_pt, h_right_pt = center - np.array([h_axis, 0]), center + np.array([h_axis, 0])
        v_bot_pt, v_top_pt = center - np.array([0, v_axis]), center + np.array([0, v_axis])

        points = np.vstack((h_left_pt, h_right_pt, v_bot_pt, v_top_pt))

        if difficulty > 1:
            points = rotate(points, ang)  # Returns rotated end-of-axis points
        if np.any(points < 0) or np.any(points > IMG_DIM):  # TODO check if this works as intended
            end = time.time()
            total_errors += 1
            total_time_in_errors += end - start
            continue

        # print(points)
        # Color in BGR
        color = 120  # TODO: Determine how to select color for shapes based on background and other shape colors if they exist on the image

        # Line thickness of -1 px
        thickness = -1
        # print(f"Errors in ellipse {total_errors}\nTime in errors {total_time_in_errors}")
        return center, h_axis, v_axis, ang, startAngle, endAngle, color, thickness, points, total_time_in_errors, total_errors


def generate_dataset(dataset_size=1000, path_folder="generated_images/dataset1/"):
    Path(path_folder).mkdir(parents=True, exist_ok=True)
    difficulties = [1, 2, 3, 4]
    functions = [generate_triangle, generate_square,
                 generate_ellipse
                 ]
    label_map = {
        generate_triangle: "triangle",
        generate_square: "square",
        generate_ellipse: "ellipse",
    }

    prod = list(islice(cycle(product(functions, difficulties)), dataset_size))
    error_time = 0
    total_errors = 0
    for i, pair in enumerate(prod):
        img = np.zeros((IMG_DIM, IMG_DIM), dtype='uint8')
        while True:
            bg_color = random.randint(0, 255)
            fig_color = random.randint(0, 255)
            if abs(bg_color-fig_color) > COLOR_GRADIENT_THRESHOLD:
                break

        img += bg_color

        func, diff = pair
        points = func(diff)

        if func == generate_ellipse:
            center, h_axis, v_axis, ang, startAngle, endAngle, color, thickness, points, t, e = func(diff)
            error_time += t
            total_errors += e
            cv2.ellipse(img, center, (h_axis, v_axis), ang,
                        startAngle, endAngle, fig_color, thickness)
        else:
            cv2.fillPoly(img, pts=[points], color=fig_color
            )

        if diff > 2:
            line_points = get_overlapping_line(points)
            cv2.line(img, line_points[0], line_points[1], 255, 1)

        if diff > 3:
            img = noisy(img, 'poisson')

        label = label_map[func]
        class_dir = os.path.join(path_folder, label)
        Path(class_dir).mkdir(exist_ok=True)

        cv2.imwrite(os.path.join(class_dir, f"{label}_{str(i)}_diff{diff}.png"), img)
        # TODO create file that maps image names to labels
    print(f"Total errors {total_errors}\nTotal time in ellipse errors {error_time}")


def generate_by_difficulty(dataset_size=250, path_folder="generated_images/dataset1/", difficulty=1):
    Path(path_folder).mkdir(parents=True, exist_ok=True)
    functions = [generate_triangle, generate_square,
                 generate_ellipse
                 ]
    label_map = {
        generate_triangle: "triangle",
        generate_square: "square",
        generate_ellipse: "ellipse",
    }
    diff_map = {
        1: 'easy',
        2: 'medium_easy',
        3: 'medium_hard',
        4: 'hard',
    }

    error_time = 0
    total_errors = 0
    loop_ctrl = list(islice(cycle(functions), dataset_size))
    for i, func in enumerate(loop_ctrl):
        img = np.zeros((IMG_DIM, IMG_DIM), dtype='uint8')
        while True:
            bg_color = random.randint(0, 255)
            fig_color = random.randint(0, 255)
            if abs(bg_color-fig_color) > COLOR_GRADIENT_THRESHOLD:
                break

        img += bg_color

        points = func(difficulty)

        if func == generate_ellipse:
            center, h_axis, v_axis, ang, startAngle, endAngle, color, thickness, points, t, e = func(difficulty)
            error_time += t
            total_errors += e
            cv2.ellipse(img, center, (h_axis, v_axis), ang,
                        startAngle, endAngle, fig_color, thickness)
        else:
            cv2.fillPoly(img, pts=[points], color=fig_color
            )

        if difficulty > 2:
            line_points = get_overlapping_line(points)
            cv2.line(img, line_points[0], line_points[1], 255, 1)

        if difficulty > 3:
            img = noisy(img, 'poisson')

        difficulty_label = diff_map[difficulty]
        shape_label = label_map[func]
        class_dir = os.path.join(path_folder, difficulty_label, shape_label)
        Path(class_dir).mkdir(parents=True, exist_ok=True)
        img_path = os.path.join(class_dir, f"{shape_label}_{str(i)}_diff{difficulty}.png")
        print(f'Saving of shape {shape_label} and difficulty {difficulty_label} image at: {img_path}')
        cv2.imwrite(img_path, img)
        # TODO create file that maps image names to labels
    print(f"Total errors {total_errors}\nTotal time in ellipse errors {error_time}")



def main():
    # img = np.zeros((IMG_DIM, IMG_DIM), dtype='uint8')

    # img += 40

    # shape = get_shape()
    # difficulty = get_difficulty_level()
    # if shape == 'triangle':
    #     points = generate_triangle(difficulty)
    # elif shape == 'quad':
    #     points = generate_square(difficulty)
    # elif shape == 'ellipse':
    #     generate_ellipse()
    #
    # cv2.fillPoly(img, pts=[points], color=(144))

    # center,h_axis, v_axis, ang, startAngle, endAngle, color, thickness = generate_ellipse()
    # importstring = cv2.ellipse(img, center, (h_axis, v_axis), ang,
    #                           startAngle, endAngle, color, thickness)

    # if difficulty > 2:
    #     line_points = get_overlapping_line(points)
    #     cv2.line(img, line_points[0], line_points[1], 255, 3)
    #
    # if difficulty > 3:
    #     img = noisy(img, 'poisson')

    # cv2.imshow("nesto", img)
    #
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
    # generate_dataset(DATASET_SIZE, DATASET_BASE_DIR)
    for i in range(1, 5):
        generate_by_difficulty(30, ".\\generated_images\\dataset2\\", i)
    # train, test = load_dataset(DATASET_BASE_DIR)


if __name__ == '__main__':
    main()
