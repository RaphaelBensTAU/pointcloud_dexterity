from sdf import *
import numpy as np

MIN_LENGTH = 0.05
MAX_LENGTH = 0.1

MIN_WIDTH = 0.05
MAX_WIDTH = 0.1

MIN_HEIGHT = 0.05
MAX_HEIGHT = 0.1

for k in range(100):
    box_length = np.random.uniform(MIN_LENGTH, MAX_LENGTH)
    box_width = np.random.uniform(MIN_WIDTH, MAX_WIDTH)
    box_height = np.random.uniform(MIN_HEIGHT, MAX_HEIGHT)
    k_box = box((box_length, box_width, box_height))
    k_box.save(f'../../meshes/box/box_{k}.stl', step = 0.005)
    with open(f'../../meshes/box/box_{k}.txt', 'w+') as f:
        f.write(f'{box_length},{box_width},{box_height}')
