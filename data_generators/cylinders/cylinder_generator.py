from sdf import *
import numpy as np

MIN_CYLINDER_RADIUS = 0.02
MAX_CYLINDER_RADIUS = 0.04

MIN_CYLINDER_HEIGHT = 0.06
MAX_CYLINDER_HEIGHT = 0.14

MIN_TORUS_RADIUS = 0.001
MAX_TORUS_RADIUS = 0.005

MAX_NUM_TORUS = 3
MIN_NUM_TORUS = 0

for k in range(100):
    print(k)
    cylinder_radius = np.random.uniform(MIN_CYLINDER_RADIUS, MAX_CYLINDER_RADIUS)
    cylinder_height = np.random.uniform(MIN_CYLINDER_HEIGHT, MAX_CYLINDER_HEIGHT)
    cylinder = rounded_cylinder(cylinder_radius, 0., cylinder_height)
    num_torus = np.random.randint(MIN_NUM_TORUS, MAX_NUM_TORUS + 1)
    torus_array = []

    for i in range(num_torus):
        torus_radius = np.random.uniform(MIN_TORUS_RADIUS, MAX_TORUS_RADIUS)
        torus_height = np.random.uniform(-cylinder_height / 2, cylinder_height / 2)
        torus_i = torus(cylinder_radius, torus_radius)
        torus_i = torus_i.translate((0, 0, torus_height))
        cylinder |= torus_i.k(0.01)
    cylinder.save(f'../../meshes/cylinders/cylinder_{k}.stl', step = 0.005)
    with open(f'../../meshes/cylinders/cylinder_{k}.txt', 'w+') as f:
        f.write(f'{cylinder_height},{cylinder_radius}')
