import parse_data.prepare_data as prepare_data
import globals
from plotting.plot_octagon import calculate_coordinates, calculate_wall_thirds_coordinates, generate_alcove_endpoints, concatenate_all_coord_lists, get_zipped_vertex_coordinates, plot_octagon_from_coords, plot_all_octagon_coordinates, return_octagon_path_points, plot_octagon
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

# default arguments
num_walls = 8
diameter = 36.21
radius = diameter/2
angle_sep = 2*math.pi/num_walls
# alcoves
alcove_length_scaled = 3.0
alcove_length_axis_projection = (alcove_length_scaled*math.sin(math.pi/4))/math.sin(math.pi/2)
                               
vertex_x, vertex_y = calculate_coordinates(vertex=True)
wall_thirds_x, wall_thirds_y = calculate_wall_thirds_coordinates(vertex_x, vertex_y)
alcove_x, alcove_y = generate_alcove_endpoints(wall_thirds_x, wall_thirds_y, alcove_length_scaled, alcove_length_axis_projection)

print("Alcove X Coordinates:", alcove_x)
print("Alcove Y Coordinates:", alcove_y)

# Coordinates dictionary

alcove_coordinates = {}
num_walls = 8

# Assign coordinates to walls
for i in range(num_walls):
    alcove_coordinates[i + 1] = (
      (alcove_x[i * 2], alcove_x[i * 2 + 1]), 
        (alcove_y[i * 2], alcove_y[i * 2 + 1])
    )

for wall, coords in alcove_coordinates.items():
    print(f"Wall {wall}: X Coordinates: {coords[0]}, Y Coordinates: {coords[1]}")

