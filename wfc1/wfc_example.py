# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:08:10 2019

@author: Isaac
"""

import matplotlib

# matplotlib.use('agg')

import types
import math
from IPython.core.debugger import set_trace
import collections

import matplotlib.pyplot as plt

# from matplotlib.pyplot import figure, subplot, subplots, title, matshow


import wfc_utilities
from wfc_utilities import CoordXY, CoordRC, hash_downto, find_pattern_center
from wfc_utilities import WFC_PARTIAL_BLANK, WFC_NULL_VALUE

import wfc_tiles
from wfc_tiles import (
    load_source_image,
    image_to_tiles,
    tiles_to_images,
    make_tile_catalog,
    show_input_to_output,
    show_extracted_tiles,
    show_false_color_tile_grid,
)

import wfc_patterns
from wfc_patterns import (
    make_pattern_catalog_no_rotations,
    render_pattern,
    show_pattern_catalog,
)

from wfc_adjacency import (
    adjacency_extraction_consistent,
    is_valid_overlap_xy,
    blit,
    show_adjacencies,
)

import wfc_solver
from wfc_solver import (
    wfc_init,
    wfc_run,
    show_wfc_patterns,
    show_pattern_adjacency,
    visualize_propagator_matrix,
    visualize_entropy,
    wfc_clear,
    visualize_compatible_count,
    show_rendered_patterns,
    render_patterns_to_output,
    wfc_observe,
    wfc_partial_output,
    wrap_coords,
    show_crystal_time,
)
import logging
import numpy as np

print(f"Using numpy version {np.__version__}")

np.set_printoptions(threshold=np.inf)

##
## Set up the namespace
##


wfc_ns_chess = types.SimpleNamespace(
    output_path="output/",
    img_filename="samples/Red Maze.png",  # name of the input file
    output_file_number=40,
    seed=587386,  # seed for random generation, can be any number
    tile_size=1,  # size of tile, in pixels
    pattern_width=2,  # Size of the patterns we want. 2x2 is the minimum, larger scales get slower fast.
    channels=3,  # Color channels in the image (usually 3 for RGB)
    adjacency_directions=dict(
        enumerate(
            [
                CoordXY(x=0, y=-1),
                CoordXY(x=1, y=0),
                CoordXY(x=0, y=1),
                CoordXY(x=-1, y=0),
            ]
        )
    ),  # The list of adjacencies that we care about - these will be turned into the edges of the graph
    periodic_input=True,  # Does the input wrap?
    periodic_output=True,  # Do we want the output to wrap?
    generated_size=(16, 8),  # Size of the final image
    screenshots=1,  # Number of times to run the algorithm, will produce this many distinct outputs
    iteration_limit=0,  # After this many iterations, time out. 0 = never time out.
    allowed_attempts=1,
)  # Give up after this many contradictions

wfc_ns_chess = find_pattern_center(wfc_ns_chess)

wfc_ns_chess = wfc_utilities.load_visualizer(wfc_ns_chess)

##
## Load image and make tile data structures
##

wfc_ns_chess.img = load_source_image(wfc_ns_chess.img_filename)
wfc_ns_chess.tiles = image_to_tiles(wfc_ns_chess.img, wfc_ns_chess.tile_size)
(
    wfc_ns_chess.tile_catalog,
    wfc_ns_chess.tile_grid,
    wfc_ns_chess.code_list,
    wfc_ns_chess.unique_tiles,
) = make_tile_catalog(wfc_ns_chess)
wfc_ns_chess.tile_ids = {
    v: k for k, v in dict(enumerate(wfc_ns_chess.unique_tiles[0])).items()
}
print(wfc_ns_chess.unique_tiles)
wfc_ns_chess.tile_weights = {
    a: b for a, b in zip(wfc_ns_chess.unique_tiles[0], wfc_ns_chess.unique_tiles[1])
}
print(wfc_ns_chess.tile_weights)

# print("wfc_ns_chess.tile_catalog")
# print(wfc_ns_chess.tile_catalog)
# print("wfc_ns_chess.tile_grid")
# print(wfc_ns_chess.tile_grid)
# print("wfc_ns_chess.code_list")
# print(wfc_ns_chess.code_list)
# print("wfc_ns_chess.unique_tiles")
# print(wfc_ns_chess.unique_tiles)

# assert False

show_input_to_output(wfc_ns_chess)
show_extracted_tiles(wfc_ns_chess)
show_false_color_tile_grid(wfc_ns_chess)

# im = np.array([[[255, 0, 0], [255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0]],
#               [[255, 0, 0], [255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0]],
#               [[255, 0, 0], [255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0]],
#               [[255, 0, 0], [255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0]],
#               [[255, 0, 0], [255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0]],
#               [[255, 0, 0], [255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0]],
#               [[255, 0, 0], [255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0]]])

# plt.imshow(im)
# plt.show()


# def adjacency_extraction_consistent(wfc_ns, pattern_cat):
#    """Takes a pattern catalog, returns a list of all legal adjacencies."""
#    # This is a brute force implementation. We should really use the adjacency list we've already calculated...
#    legal = []
#    #print(f"pattern_cat\n{pattern_cat}")
#    for p1, pattern1 in enumerate(pattern_cat):
#        for d_index, d in enumerate(wfc_ns.adjacency_directions):
#            for p2, pattern2 in enumerate(pattern_cat):
#                if is_valid_overlap_xy(d, p1, p2, pattern_cat, wfc_ns.pattern_width, wfc_ns.adjacency_directions):
#                    legal.append((d_index, p1, p2))
#    return legal

# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput
# with PyCallGraph(output=GraphvizOutput()):


##
## Patterns
##

(
    wfc_ns_chess.pattern_catalog,
    wfc_ns_chess.pattern_weights,
    wfc_ns_chess.patterns,
) = make_pattern_catalog_no_rotations(
    wfc_ns_chess.tile_grid, wfc_ns_chess.pattern_width
)
show_pattern_catalog(wfc_ns_chess)
adjacency_relations = adjacency_extraction_consistent(
    wfc_ns_chess, wfc_ns_chess.patterns
)
# print(adjacency_relations)
show_adjacencies(wfc_ns_chess, adjacency_relations[:256])
# print(wfc_ns_chess.patterns)


##
## Run the solver
##

solution = wfc_run(wfc_ns_chess, visualize=False)

##
## Output the results
##

import moviepy.editor as mpy
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

# from lucid_serialize_array import _normalize_array

solution = wfc_solver.render_recorded_visualization(solution)

video_fn = f"{solution.wfc_ns.output_path}/crystal_example_{solution.wfc_ns.output_file_number}.mp4"
print("*****************************")
print(solution.solver_recording_stack[0].shape)
writer = FFMPEG_VideoWriter(video_fn, solution.solver_recording_stack[0].shape, 12.0)
# for i in range(24):
#    writer.write_frame(solution.solver_recording_stack[0])
for img_data in solution.solver_recording_stack:
    writer.write_frame(img_data)
    # print(_normalize_array(img_data))
    print("!", end="")
# for i in range(24):
#    writer.write_frame(solution.solver_recording_stack[-1])
# for i in range(24):
#    writer.write_frame(solution.solver_recording_stack[0])
# for img_data in solution.solver_recording_stack:
#    writer.write_frame(img_data)
#    #print(_normalize_array(img_data))
#    print('!',end='')
# for i in range(24):
#    writer.write_frame(solution.solver_recording_stack[-1])

writer.close()

mpy.ipython_display(video_fn, height=700)

solution = wfc_partial_output(solution)
show_rendered_patterns(solution, True)
render_patterns_to_output(solution, True)
# show_crystal_time(solution, True)
# show_rendered_patterns(solution, False)
# render_patterns_to_output(solution, False)
