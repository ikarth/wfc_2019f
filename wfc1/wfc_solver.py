# ## Current WFC Solver

# In[22]:

import matplotlib
#matplotlib.use('Agg')

import types
from wfc.wfc_adjacency import adjacency_extraction_consistent
import numpy as np
from wfc.wfc_utilities import WFC_PARTIAL_BLANK, WFC_NULL_VALUE
import matplotlib.pyplot
from matplotlib.pyplot import figure, subplot, subplots, title, matshow
from wfc.wfc_patterns import render_pattern
from wfc.wfc_adjacency import blit
from wfc.wfc_tiles import tiles_to_images
import wfc.wfc_utilities
from wfc.wfc_utilities import CoordXY, CoordRC, hash_downto, find_pattern_center
import random
import copy
import time

import imageio

import logging
logging.basicConfig(level=logging.INFO)
wfc_logger = logging.getLogger()

import math

import pdb

#import moviepy.editor as mpy
#from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter



WFC_DEBUGGING = False
WFC_VISUALIZE = False

WFC_FINISHED = -2
WFC_FAILURE  = -1
WFC_TIMEDOUT = -3
WFC_FAKE_FAILURE = -6

### Visualization Functions

# In[23]:


#def pattern_to_tile(pattern, pattern_catalog, pattern_center):
#    try:
#        return pattern_catalog[pattern][pattern_center]
#    except:
#        return pattern_catalog[0][pattern_center]


# In[24]:


def status_print_helper(status_string):
    #print(status_string)
    pass

# In[25]:


def show_wfc_patterns(wfc_state, pattern_translations):
    s_columns = 24 // min(24, wfc_state.wfc_ns.pattern_width)
    s_rows = 1 + (int(len(pattern_translations)) // s_columns)
    fig = figure(figsize=(32, 32))

    title('Extracted Patterns')
    for i,tcode in enumerate(pattern_translations):
        pat_cat = pattern_translations[i]
        ptr = render_pattern(pat_cat, wfc_state.wfc_ns).astype(np.uint8)
        sp = subplot(s_rows, s_columns, i+1)
        spi = sp.imshow(ptr)
        spi.axes.xaxis.set_label_text(f'({wfc_state.wfc_ns.pattern_weights[i]})')
        spi.axes.tick_params(labelleft=False,labelbottom=False, left=False, bottom=False)
        spi.axes.grid(color="grey", linewidth=1.0)
        for axis in [spi.axes.xaxis, spi.axes.yaxis]:
            axis.set_ticks(np.arange(-0.5, wfc_state.wfc_ns.pattern_width + 0.5, 1))
        sp.set_title(i)
    matplotlib.pyplot.close(fig)

# In[26]:


def visualize_propagator_matrix(p_matrix):
    visual_stack = np.empty([p_matrix.shape[1], p_matrix.shape[2]], dtype=p_matrix.dtype)
    visual_stack = (p_matrix[0] * 1) + (p_matrix[1] * 2) + (p_matrix[2] * 4) + (p_matrix[3] * 8)
    matfig = figure(figsize=(9,9))
 
    matfig.tight_layout(pad=0)
    ax = subplot(1,1,1)

    title('Propagator Matrix')
    #ax.matshow(visual_stack,cmap='jet',fignum=matfig.number)
    ax.matshow(visual_stack,cmap='jet')
    ax.grid(None)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(None)
    #ax.grid(color="black", linewidth=1.5)
    #for axis in [ax.xaxis, ax.yaxis]:
    #        axis.set_ticks(np.arange(-0.5, p_matrix.shape[1] + 0.5, 1.0))
    matplotlib.pyplot.close(matfig)

def record_visualization(wfc_state, wfc_vis = None):
    if(None == wfc_vis):
        wfc_vis = types.SimpleNamespace(
            method_time_stack = [],
            ones_time_stack = [],
            output_time_stack = [],
            partial_output_time_stack = [],
            crystal_time_stack = [],
            choices_recording_stack =[],
            number_of_patterns = wfc_state.number_of_patterns,
            pattern_center = wfc_state.wfc_ns.pattern_center,
            rows = wfc_state.rows,
            columns = wfc_state.columns,
            pattern_catalog = wfc_state.wfc_ns.pattern_catalog,
            wfc_ns = copy.deepcopy(wfc_state.wfc_ns),
            wave_table_stack = [],
            solver_recording_stack = [],
        )
    wfc_vis.method_time_stack.append(np.copy(wfc_state.method_time))
    wfc_vis.ones_time_stack.append(np.copy(np.count_nonzero(wfc_state.wave_table, axis=2)))
    wfc_vis.output_time_stack.append(np.copy(wfc_state.output_grid))
    wfc_vis.partial_output_time_stack.append(np.copy(wfc_state.partial_output_grid))
    wfc_vis.crystal_time_stack.append(np.copy(wfc_state.crystal_time))
    wfc_vis.choices_recording_stack.append(np.copy(wfc_state.choices_recording))
    wfc_vis.wave_table_stack.append(np.copy(wfc_state.wave_table))
    print(f"time stack length: {len(wfc_vis.method_time_stack)}")
    return wfc_vis

def render_recorded_visualization(wfc_vis):
    #wfc_vis = wfc_solution_to_vis.recorded_vis

    wfc_logger.info(f"time stack length: {len(wfc_vis.method_time_stack)}")
    wfc_logger.info(f"method time stack: {[x.sum() for x in wfc_vis.method_time_stack]}")
    for i in range(len(wfc_vis.method_time_stack)):
        matfig = figure(figsize=(16,8))

        #matplotlib.pyplot.title(f"{wfc_state.wfc_ns.output_file_number}_{backtrack_track_global}_{wfc_state.current_iteration_count_last_touch}", fontsize=14, fontweight='bold', y = -1)
        matplotlib.pyplot.title(f"{i}", fontsize=14, fontweight='bold', y = 0.6)
        
        ax = subplot(1,5,1)
        title('Resolution Method')
        ax.matshow(wfc_vis.method_time_stack[i],cmap='magma')
        ax.grid(None)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid(None)

        ax = subplot(1,5,2)
        title('Ones Matrix')
    
        ax.matshow(wfc_vis.ones_time_stack[i],cmap='plasma',vmin=0, vmax=wfc_vis.number_of_patterns)
        ax.grid(None)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid(None)
        
        #ax = subplot(1,5,3)
        #title('Output Matrix')
        
        #ax.matshow(wfc_state.output_time_stack[i],cmap='inferno', vmin=0, vmax=wfc_state.number_of_patterns)
        #ax.grid(None)
        #ax.set_yticklabels([])
        #ax.set_xticklabels([])
        #ax.grid(None)
        
        ax = subplot(1,5,3)
        title('Choices Matrix')
        
        ax.matshow(wfc_vis.choices_recording_stack[i],cmap='inferno', vmin=0, vmax=wfc_vis.number_of_patterns)
        ax.grid(None)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid(None)
    
        ax = subplot(1,5,4)
        title('Crystal Matrix')
        
        ax.matshow(wfc_vis.crystal_time_stack[i], cmap='gist_rainbow', vmin=0, vmax=len(wfc_vis.crystal_time_stack))
        ax.grid(None)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid(None)
        
        pattern_grid = np.array(wfc_vis.output_time_stack[i], dtype=np.int64)
        
        has_gaps = np.any(np.count_nonzero(wfc_vis.wave_table_stack[i], axis=2) != 1) 
        if has_gaps:
            pattern_grid = np.array(wfc_vis.partial_output_time_stack[i], dtype=np.int64)
        render_grid = np.full(pattern_grid.shape,  WFC_PARTIAL_BLANK, dtype=np.int64)
        pattern_center = wfc_vis.wfc_ns.pattern_center
        for row in range(wfc_vis.rows):
            for column in range(wfc_vis.columns):
                if (len(pattern_grid.shape) > 2):
                    pattern_list = []
                    for z in range(wfc_vis.number_of_patterns):
                        pattern_list.append(pattern_grid[(row,column,z)])
                    pattern_list = [pattern_grid[(row,column,z)] for z in range(wfc_vis.number_of_patterns) if (pattern_grid[(row,column,z)] != -1) and (pattern_grid[(row,column,z)] != WFC_NULL_VALUE)]
                    for pl_count, the_pattern in enumerate(pattern_list):
                        the_pattern_tiles = wfc_vis.pattern_catalog[the_pattern][pattern_center[0]:pattern_center[0]+1,pattern_center[1]:pattern_center[1]+1]
                        render_grid = blit(render_grid, the_pattern_tiles, (row,column), layer = pl_count)
                else:
                    if WFC_NULL_VALUE != pattern_grid[(row,column)]:
                        the_pattern = wfc_vis.wfc_ns.pattern_catalog[pattern_grid[(row,column)]]
                        p_x = wfc_vis.wfc_ns.pattern_center[0]
                        p_y = wfc_vis.wfc_ns.pattern_center[1]
                        the_pattern = the_pattern[p_x:p_x+1, p_y:p_y+1]
                        render_grid = blit(render_grid, the_pattern, (row, column))
        ptr = tiles_to_images(wfc_vis.wfc_ns, render_grid, wfc_vis.wfc_ns.tile_catalog, wfc_vis.wfc_ns.tile_size, visualize=True, partial=True).astype(np.uint8)
        
        ax = subplot(1,5,5)
        title('Output Matrix')
        
        ax.imshow(ptr)
        ax.grid(None)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid(None)
        
        matplotlib.pyplot.savefig(f'{wfc_vis.wfc_ns.output_path}crystal_preview_{wfc_vis.wfc_ns.output_file_number}_{backtrack_track_global}_{i}_{str(time.time())}.png', bbox_inches='tight')
        
        img_data = np.frombuffer(matfig.canvas.tostring_rgb(), dtype=np.uint8)
        img_data = img_data.reshape(matfig.canvas.get_width_height() + (3,))

        #print(f"img_data shape: {matfig.canvas.get_width_height()} {matfig.canvas.get_width_height()[::-1] + (3,)}")

        # temporarily disable the recording stack...
        #wfc_vis.solver_recording_stack.append(img_data)
        matplotlib.pyplot.close(fig=matfig)
    wfc_logger.info(f"recording stack length: {len(wfc_vis.solver_recording_stack)}")
    return wfc_vis

def visualize_entropies(wfc_state):
    matfig = figure(figsize=(24,24))

    matplotlib.pyplot.title(f"{wfc_state.wfc_ns.output_file_number}_{backtrack_track_global}_{wfc_state.current_iteration_count_last_touch}", fontsize=14, fontweight='bold', y= 0.6)

    
    ax = subplot(1,5,1)
    title('Resolution Method')
    for row in range(wfc_state.rows):
        for column in range(wfc_state.columns):
            try:
                entropy_sum = wfc_state.sums_of_weights[row, column]
                wfc_state.entropies[row, column] = (math.log(entropy_sum)) - ((wfc_state.sums_of_weight_log_weights[row, column]) / entropy_sum)
            except:
                pass
    
    ax.matshow(wfc_state.method_time,cmap='magma')
    ax.grid(None)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(None)
    
    ax = subplot(1,5,2)
    title('Ones Matrix')
    
    ax.matshow(np.count_nonzero(wfc_state.wave_table, axis=2),cmap='plasma',vmin=0, vmax=wfc_state.number_of_patterns)
    ax.grid(None)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(None)

#    ax = subplot(1,5,3)
#    title('Output Matrix')
#    
#    ax.matshow(wfc_state.output_grid,cmap='inferno', vmin=0, vmax=wfc_state.number_of_patterns)
#    ax.grid(None)
#    ax.set_yticklabels([])
#    ax.set_xticklabels([])
#    ax.grid(None)
    
    ax = subplot(1,5,3)
    title('Count of Choices')
    
    ax.matshow(wfc_state.choices_recording,cmap='magma', vmin=0, vmax=math.log(wfc_state.number_of_patterns))
    ax.grid(None)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(None)

    ax = subplot(1,5,4)
    title('Crystal Matrix')
    
    ax.matshow(wfc_state.crystal_time % 512.0,cmap='gist_rainbow', vmin=0, vmax=512)
    ax.grid(None)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(None)
    
    pattern_grid = np.array(wfc_state.output_grid, dtype=np.int64)
    
    has_gaps = np.any(np.count_nonzero(wfc_state.wave_table, axis=2) != 1) 
    if has_gaps:
        pattern_grid = np.array(wfc_state.partial_output_grid, dtype=np.int64)
    render_grid = np.full(pattern_grid.shape,  WFC_PARTIAL_BLANK, dtype=np.int64)
    pattern_center = wfc_state.wfc_ns.pattern_center
    for row in range(wfc_state.rows):
        for column in range(wfc_state.columns):
            if (len(pattern_grid.shape) > 2):
                pattern_list = []
                for z in range(wfc_state.number_of_patterns):
                    pattern_list.append(pattern_grid[(row,column,z)])
                pattern_list = [pattern_grid[(row,column,z)] for z in range(wfc_state.number_of_patterns) if (pattern_grid[(row,column,z)] != -1) and (pattern_grid[(row,column,z)] != WFC_NULL_VALUE)]
                for pl_count, the_pattern in enumerate(pattern_list):
                    the_pattern_tiles = wfc_state.wfc_ns.pattern_catalog[the_pattern][pattern_center[0]:pattern_center[0]+1,pattern_center[1]:pattern_center[1]+1]
                    render_grid = blit(render_grid, the_pattern_tiles, (row,column), layer = pl_count)
            else:
                if WFC_NULL_VALUE != pattern_grid[(row,column)]:
                    the_pattern = wfc_state.wfc_ns.pattern_catalog[pattern_grid[(row,column)]]
                    p_x = wfc_state.wfc_ns.pattern_center[0]
                    p_y = wfc_state.wfc_ns.pattern_center[1]
                    the_pattern = the_pattern[p_x:p_x+1, p_y:p_y+1]
                    render_grid = blit(render_grid, the_pattern, (row, column))
    ptr = tiles_to_images(wfc_state.wfc_ns, render_grid, wfc_state.wfc_ns.tile_catalog, wfc_state.wfc_ns.tile_size, visualize=True, partial=True).astype(np.uint8)
    
    #ax.grid(color="magenta", linewidth=1.5)
    #ax.tick_params(direction='in', bottom=False, left=False)

    #for axis, dim in zip([ax.xaxis, ax.yaxis],[wfc_state.columns, wfc_state.rows]):
    #    axis.set_ticks(np.arange(-0.5, dim + 0.5, 1))
    #    axis.set_ticklabels([])

    
    ax = subplot(1,5,5)
    title('Output Matrix')
    
    ax.imshow(ptr)
    ax.grid(None)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(None)

    
    matplotlib.pyplot.savefig(f'{wfc_state.wfc_ns.output_path}crystal_preview_{wfc_state.wfc_ns.output_file_number}_{backtrack_track_global}_{wfc_state.current_iteration_count_last_touch}.png', bbox_inches='tight')
    
    img_data = np.frombuffer(matfig.canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(matfig.canvas.get_width_height() + (3,))
    #print(f"img_data shape: {matfig.canvas.get_width_height()} {matfig.canvas.get_width_height()[::-1] + (3,)}")
    matplotlib.pyplot.close(matfig)
    #matplotlib.clear()
    return img_data
    
    
    

# In[27]:


def show_pattern_adjacency(wfc_state):
    s_columns = 4#wfc_state.number_of_directions // 2
    s_rows = 1#int(wfc_state.number_of_directions % 2)
    cat_size = (len(wfc_state.wfc_ns.pattern_catalog) + 1)
    pat_adj_size = wfc_state.wfc_ns.pattern_width * cat_size
    
    fig = figure(figsize=(s_columns * 7.0, s_rows * 7.0))

    title('Pattern Adjacency')
    for d_index, d_offset in wfc_state.wfc_ns.adjacency_directions.items():
    
        adj_preview = np.full((pat_adj_size,pat_adj_size), -1, dtype=np.int64)
        for x in range(cat_size):
            for y in range(cat_size):
                the_pattern = None
                if (0 == y) and x > 0:
                    the_pattern = wfc_state.wfc_ns.pattern_catalog[x-1]
                if (0 == x) and y > 0:
                    the_pattern = wfc_state.wfc_ns.pattern_catalog[y-1]
                if (x > 0) and (y > 0):
                    if wfc_state.propagator_matrix[d_index, y-1, x-1]:
                        the_pattern = np.array([[-2,-2],[-2,-2]], dtype=np.int64)
                if type(None) != type(the_pattern):
                    adj_preview = blit(adj_preview, the_pattern, (x*wfc_state.wfc_ns.pattern_width,y*wfc_state.wfc_ns.pattern_width))
        ptr = tiles_to_images(wfc_state.wfc_ns, adj_preview, wfc_state.wfc_ns.tile_catalog, wfc_state.wfc_ns.tile_size, visualize=True).astype(np.uint8)
        ax = subplot(s_rows, s_columns, 1+d_index)
        ax.grid(color="magenta", linewidth=1.5)
        im = ax.imshow(ptr)
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_ticks(np.arange(-0.5, pat_adj_size + 0.5, wfc_state.wfc_ns.pattern_width))
        title("Direction {}\nr,c({})\nor x,y({})".format(d_index, wfc_state.wfc_ns.adjacency_directions_rc[d_index], d_offset), fontsize=15)
    matplotlib.pyplot.close(fig)

# In[28]:


import itertools
import math



def weight_log(val):
    return val * math.log(val)

def wfc_init(prestate):
    prestate.adjacency_directions_rc = {i:CoordRC(a.y, a.x) for i,a in prestate.adjacency_directions.items()}
    prestate = wfc.wfc_utilities.find_pattern_center(prestate)
    wfc_state = types.SimpleNamespace(wfc_ns = prestate)

    wfc_state.fake_failure = False
    
    wfc_state.result = None
    wfc_state.adjacency_relations = adjacency_extraction_consistent(wfc_state.wfc_ns, wfc_state.wfc_ns.patterns)
    if WFC_DEBUGGING:
        wfc_logger.debug(f"wfc_state.adjacency_relations:\n{wfc_state.adjacency_relations}")
    #    status_print_helper(f"wfc_state.wfc_ns.patterns {wfc_state.wfc_ns.patterns}")
    
    wfc_logger.debug("wfc_init():patterns")
    
    wfc_state.patterns = np.array(list(wfc_state.wfc_ns.pattern_catalog.keys()))
    wfc_state.pattern_translations = list(wfc_state.wfc_ns.pattern_catalog.values())
    wfc_state.number_of_patterns = wfc_state.patterns.size
    if WFC_DEBUGGING:
        wfc_logger.debug("number_of_patterns: {}".format(wfc_state.number_of_patterns))
        wfc_logger.debug("patterns: {}".format(wfc_state.patterns))
        wfc_logger.debug("pattern translations: {}".format(wfc_state.pattern_translations))
    if WFC_VISUALIZE:
        show_wfc_patterns(wfc_state, wfc_state.pattern_translations)
    
    wfc_state.number_of_directions = len(wfc_state.wfc_ns.adjacency_directions)
    #wfc_state.reverse_adjacency_directions = make_reverse_adjacency_directions(wfc_state.wfc_ns.adjacency_directions)
    #if WFC_DEBUGGING:
    #    status_print_helper("reverse_adjacency_directions: {}".format(wfc_state.reverse_adjacency_directions))
    
    # The Propagator is a data structure that holds the adjacency information 
    # for the patterns, i.e. given a direction, which patterns are allowed to 
    # be placed next to the pattern that we're currently concerned with.
    # This won't change over the course of using the solver, so the important
    # thing here is fast lookup.
    wfc_state.propagator_matrix = np.zeros((wfc_state.number_of_directions, 
                                            wfc_state.number_of_patterns, 
                                            wfc_state.number_of_patterns), dtype=np.bool_)
    
    wfc_logger.debug("wfc_init():adjacency_relations")
    
    # While the adjacencies were stored as (x,y) pairs, we're going to use (row,column) pairs here.
    #wfc_state.reversed_directions = [(r,c) for c,r in wfc_state.wfc_ns.adjacency_directions.values()]
    #print(f"wfc_state.reversed_directions:\n{wfc_state.reversed_directions}")
    for d,p1,p2 in wfc_state.adjacency_relations:
        wfc_state.propagator_matrix[(d, p1, p2)] = True
    
        
    if WFC_VISUALIZE:
        visualize_propagator_matrix(wfc_state.propagator_matrix)
        show_pattern_adjacency(wfc_state)
    
    # The Wave Table is the boolean expression of which patterns are allowed 
    # in which cells of the solution we are calculating.
    wfc_state.rows = wfc_state.wfc_ns.generated_size[0]
    wfc_state.columns = wfc_state.wfc_ns.generated_size[1]
    
    wfc_state.solving_time = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.int32)
    wfc_state.propagation_time = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.int32)
    
    wfc_state.wave_shape = [wfc_state.rows, wfc_state.columns, wfc_state.number_of_patterns]
    wfc_state.wave_table = np.full(wfc_state.wave_shape, True, dtype=np.bool_)
    
    # The compatible_count is a running count of the number of patterns that 
    # are still allowed to be next to this cell in a particular direction.
    compatible_shape = [wfc_state.rows, 
                        wfc_state.columns, 
                        wfc_state.number_of_patterns, 
                        wfc_state.number_of_directions]

    wfc_logger.debug(f"compatible shape:{compatible_shape}")
    wfc_state.compatible_count = np.full(compatible_shape, wfc_state.number_of_patterns, dtype=np.int16) # assumes that there are less than 65536 patterns
    
    wfc_logger.debug("wfc_init():weights")
    
    # The weights are how we manage the probabilities when we choose the next
    # pattern to place. Rather than recalculating them from scratch each time,
    # these let us incrementally update their values.
    wfc_state.weights = np.array(list(wfc_state.wfc_ns.pattern_weights.values()))
    wfc_state.weight_log_weights = np.vectorize(weight_log)(wfc_state.weights)
    if WFC_DEBUGGING:
        status_print_helper(f"wfc_state.weights {wfc_state.weights}")
        status_print_helper(f"wfc_state.weight_log_weights {wfc_state.weight_log_weights}")
    
    wfc_state.sum_of_weights = np.sum(wfc_state.weights)
    if WFC_DEBUGGING:
        status_print_helper(f"wfc_state.sum_of_weights {wfc_state.sum_of_weights}")
    wfc_state.sum_of_weight_log_weights = np.sum(wfc_state.weight_log_weights)
    wfc_state.starting_entropy = math.log(wfc_state.sum_of_weights) - (wfc_state.sum_of_weight_log_weights / wfc_state.sum_of_weights)
    
    wfc_state.entropies = np.zeros([wfc_state.rows, wfc_state.columns], dtype = np.float64)
    #wfc_state.sums_of_ones = np.zeros([wfc_state.rows, wfc_state.columns], dtype = np.float64)
    wfc_state.sums_of_weights = np.zeros([wfc_state.rows, wfc_state.columns], dtype = np.float64)
    
    # Instead of updating all of the cells for every propagation, we use a queue 
    # that marks the dirty tiles to update.
    wfc_state.observation_stack = collections.deque()
        
    wfc_state.output_grid = np.full([wfc_state.rows, wfc_state.columns], WFC_NULL_VALUE, dtype = np.int64)
    wfc_state.partial_output_grid = np.full([wfc_state.rows, wfc_state.columns, wfc_state.number_of_patterns], -9, dtype = np.int64)

    wfc_logger.debug("wfc_init():observation")

    wfc_state.current_iteration_count_observation = 0
    wfc_state.current_iteration_count_propagation = 0
    wfc_state.current_iteration_count_last_touch = 0
    wfc_state.current_iteration_count_crystal = 0
    wfc_state.solving_time = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.int32)
    wfc_state.ones_time = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.int32)
    wfc_state.propagation_time = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.int32)
    wfc_state.touch_time = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.int32)
    wfc_state.crystal_time = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.int32)
    wfc_state.method_time = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.int32)
    wfc_state.choices_recording = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.float32)

    #global recording_stack
    #recording_stack.method_time_stack = []
    #recording_stack.ones_time_stack = []
    #recording_stack.solving_time_stack = []
    #recording_stack.propagation_time_stack = []
    #recording_stack.touch_time_stack = []
    #recording_stack.crystal_time_stack = []
    #recording_stack.output_time_stack = []
    #recording_stack.solver_recording_stack = []
    #recording_stack.choices_recording_stack = []
    
    return wfc_state



# In[29]:


def visualize_entropy(wfc_state):
    matfig = figure(figsize=(7,7))

    ax = subplot(1,1,1)
    title('Sums of Weights')
    ax.matshow(wfc_state.sums_of_weights,cmap='plasma')
    ax.grid(None)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(None)
    status_print_helper(f"sum_of_weights\n{wfc_state.sums_of_weights}")
    matplotlib.pyplot.close(matfig)


# In[30]:


def wfc_clear(wfc_state):
    # Crystal solving time matrix
    wfc_state.current_iteration_count_observation = 0
    wfc_state.current_iteration_count_propagation = 0
    wfc_state.current_iteration_count_last_touch = 0
    wfc_state.current_iteration_count_crystal = 0
   
    #global recording_stack
    #recording_stack.method_time_stack = []
    #recording_stack.ones_time_stack = []
    #recording_stack.solving_time_stack = []
    #recording_stack.propagation_time_stack = []
    #recording_stack.touch_time_stack = []
    #recording_stack.crystal_time_stack = []
    #recording_stack.output_time_stack = []
    #recording_stack.solver_recording_stack = []
    #recording_stack.choices_recording_stack = []

    wfc_state.solving_time = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.int32)
    wfc_state.ones_time = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.int32)
    wfc_state.propagation_time = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.int32)
    wfc_state.touch_time = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.int32)
    wfc_state.crystal_time = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.int32)
    wfc_state.method_time = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.int32)
    wfc_state.choices_recording = np.full((wfc_state.rows, wfc_state.columns), 0, dtype=np.float32)
    
    wfc_logger.debug("reset wave table")
    # Reset the wave table.
    wfc_state.wave_table = np.full(wfc_state.wave_shape, True, dtype=np.bool_)
    
    
    compatible_shape = [wfc_state.rows, 
                        wfc_state.columns, 
                        wfc_state.number_of_patterns, 
                        wfc_state.number_of_directions]
    #wfc_state_compatible_count = np.full(compatible_shape, wfc_state.number_of_patterns, dtype=np.int16)
    wfc_logger.debug("Initialize the compatible count")
    # Initialize the compatible count from the propagation matrix. This sets the
    # maximum domain of possible neighbors for each cell node.
    #for row in range(wfc_state.rows):
    #    print(row, end=',')
    #    for column in range(wfc_state.columns):
    #        for pattern in range(wfc_state.number_of_patterns):
    #            for direction in range(wfc_state.number_of_directions):
    #                p_matrix_sum = sum(wfc_state.propagator_matrix[(direction+2)%4][pattern]) # TODO: figure out why flipping directions is needed here, maybe fix things so it isn't
    #                wfc_state_compatible_count[row, column, pattern, direction] = p_matrix_sum
#   #         #print("{},{}\n".format(row, column))
    
    def prop_compat(p,d):
        #print(p,d,end=':')
        #print(p)
        #print(d)
        #print('pm[{},{}]: {}'.format(d,p,wfc_state.propagator_matrix[d][p]))
        return sum(wfc_state.propagator_matrix[(d+2)%4][p])
    def comp_count(r,c,p,d):
        return pattern_compatible_count[p][d]
    
    pcomp = np.vectorize(prop_compat)
    ccount = np.vectorize(comp_count)
    pattern_compatible_count = np.fromfunction(pcomp, (wfc_state.number_of_patterns,wfc_state.number_of_directions), dtype=np.int16)
    wfc_state.compatible_count = np.fromfunction(ccount, (wfc_state.rows,wfc_state.columns,wfc_state.number_of_patterns,wfc_state.number_of_directions), dtype=np.int16)
    
    wfc_logger.debug("set the weights to their maximum values")
    # Likewise, set the weights to their maximum values
    #wfc_state.sums_of_ones = np.full([wfc_state.rows, wfc_state.columns],
    #                                 wfc_state.number_of_patterns,
    #                                 dtype = np.uint16)
    wfc_state.sums_of_weights = np.full([wfc_state.rows, wfc_state.columns],
                                        wfc_state.sum_of_weights, 
                                        dtype = np.float64)
    wfc_state.sums_of_weight_log_weights = np.full([wfc_state.rows, wfc_state.columns],
                                                   wfc_state.sum_of_weight_log_weights, 
                                                   dtype = np.float64)
    wfc_state.entropies = np.full([wfc_state.rows, wfc_state.columns],
                                   wfc_state.starting_entropy, 
                                   dtype = np.float64)
    if WFC_DEBUGGING:
        status_print_helper(f"starting entropy: {wfc_state.starting_entropy}")
        status_print_helper(f"wfc_state.entropies:  {wfc_state.entropies}")
        #status_print_helper(f"wfc_state.sums_of_ones: {wfc_state.sums_of_ones}")
        status_print_helper(f"wfc_state.sums_of_weights: {wfc_state.sums_of_weights}")

    wfc_state.recorded_steps = []

    wfc_state.observation_stack = collections.deque()
    # TODO: add ground-banning of patterns / masking / etc

    # ground-banning
    if wfc_state.wfc_ns.ground != 0 and False: # False => currently disabled
        for p in wfc_state.wfc_ns.pattern_catalog.keys():
            for x in range(wfc_state.rows):
                for y in range(wfc_state.columns):
                    ban_pattern = (not (p in wfc_state.wfc_ns.last_patterns) and (y >= wfc_state.wfc_ns.generated_size[1] - 1))# or ((p in wfc_state.wfc_ns.last_patterns) and (y < wfc_state.wfc_ns.generated_size[1] - 1))
                    if ban_pattern:
                        wfc_state = Ban(wfc_state, CoordRC(row=y, column=x),p)

    wfc_state.previous_decisions = []
    
    wfc_logger.debug("clear complete")

    return wfc_state



# We'll want to visualize the compatible count as the solver runs. This starts out as a uniform color (with everything at 100%) but quickly changes as individual cells start to resolve.

# In[31]:


def visualize_compatible_count(wfc_state):
    directions = wfc_state.number_of_directions
    visual_stack = np.zeros(wfc_state.compatible_count.shape[:2])
    for i in range(visual_stack.shape[0]):
        for j in range(visual_stack.shape[1]):
            for k in range(wfc_state.compatible_count.shape[2]):
                for l in range(wfc_state.compatible_count.shape[3]):
                    visual_stack[i,j] += wfc_state.compatible_count[i,j,k,l]
                    if WFC_DEBUGGING:
                        status_print_helper(f"compatible: {i},{j},{k},{l} => {wfc_state.compatible_count[i,j,k,l]}")
    matfig = figure(figsize=(7,7))

    ax = subplot(1,1,1)
    title('Compatible Count')
    ax.matshow(visual_stack,cmap='viridis')
    ax.grid(None)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(None)
    status_print_helper(f"compatible_count\n{wfc_state.compatible_count}")

    matplotlib.pyplot.close(matfig)
# In[32]:

def wfc_partial_output(wfc_state):
    wfc_state.partial_output_grid = np.full([wfc_state.rows, wfc_state.columns, wfc_state.number_of_patterns], -9, dtype = np.int64)
    
    for row in range(wfc_state.rows):
        for column in range(wfc_state.columns):
            pattern_flags = wfc_state.wave_table[row, column]
            #print(f"pattern_flags: {pattern_flags}")
            p_list = []
            for pindex, pflag in enumerate(pattern_flags):
                if pflag:
                    p_list.append(pindex)
            #print(f"p_list: {p_list}\n")
            for z,p in enumerate(p_list):
                wfc_state.partial_output_grid[row, column, z] = p
    #print(f"\n~~~ wfc_state.partial_output_grid ~~~\n{wfc_state.partial_output_grid}")
    wfc_state.recorded_steps.append(wfc_state.partial_output_grid)
    return wfc_state


# In[33]:


# A useful helper function which we use because we want numpy arrays instead of jagged arrays
# https://stackoverflow.com/questions/7632963/numpy-find-first-index-of-value-fast/7654768
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1


# In[34]:

def recalculate_weights(state, parameters, coords_index, pattern_id):
    state.sums_of_weights[coords_index.row, coords_index.column] -= parameters.weights[pattern_id]
    state.sums_of_weight_log_weights[coords_index.row, coords_index.column] -= state.weight_log_weights[pattern_id]
    entropy_sum = state.sums_of_weights[coords_index.row, coords_index.column]
    try:
        state.entropies[coords_index.row, coords_index.column] = (math.log(entropy_sum)) - ((state.sums_of_weight_log_weights[coords_index.row, coords_index.column]) / entropy_sum)
    except ValueError as e:
        logging.debug(f"Contradiction when banning {coords_index} -> {pattern_id}: {e}")
        state.result = WFC_FAILURE
    return state
    

def RecalculateWeights(wfc_state, coords_index, pattern_id):
    # uncomment to show all fails
    #if np.count_nonzero(wfc_state.wave_table[coords_index.row, coords_index.column]) < 1:
    #    wfc_logger.warning(f"Sums of ones already below 1 at {coords_index}: {wfc_state.wave_table[coords_index.row, coords_index.column].sum()}")
    
    wfc_state.sums_of_weights[coords_index.row, coords_index.column] -= wfc_state.weights[pattern_id]
    wfc_state.sums_of_weight_log_weights -= wfc_state.weight_log_weights[pattern_id]
    
    entropy_sum = wfc_state.sums_of_weights[coords_index.row, coords_index.column]
    try:
        wfc_state.entropies[coords_index.row, coords_index.column] = (math.log(entropy_sum)) - ((wfc_state.sums_of_weight_log_weights[coords_index.row, coords_index.column]) / entropy_sum)
    except ValueError as e:
        logging.debug(f"Contradiction when banning {coords_index} -> {pattern_id}: {e}")
        wfc_state.result = WFC_FAILURE
        return wfc_state
    return wfc_state

import collections
BanEntry = collections.namedtuple('BanEntry', ['coords_row', 'coords_column', 'pattern_id'])

def BanAlreadyTried(wfc_state, coords_index, pattern_id):
    wfc_state.wave_table[coords_index.row, coords_index.column, pattern_id] = False
    for direction_id, direction_offset in wfc_state.wfc_ns.adjacency_directions_rc.items():
        wfc_state.compatible_count[coords_index.row, coords_index.column, pattern_id, direction_id] = 0
    wfc_state.observation_stack.append(BanEntry(coords_index.row, coords_index.column, pattern_id))
    wfc_state = RecalculateWeights(wfc_state, coords_index, pattern_id)
    
    return wfc_state
    

def Ban(wfc_state, coords_index, pattern_id):
    if wfc_state.logging:
        with open(wfc_state.wfc_ns.debug_log_filename, "a") as stats_file:
            stats_file.write(f"Banning: {pattern_id} at {coords_index}\n")

    #pdb.set_trace()
    if wfc_state.overflow_check:
        if np.any(np.count_nonzero(wfc_state.wave_table, axis=2) > 60000):
            print("overflow E")
            print(np.count_nonzero(wfc_state.wave_table, axis=2))
            pdb.set_trace()
            assert False

    wfc_state.wave_table[coords_index.row, coords_index.column, pattern_id] = False
    for direction_id, direction_offset in wfc_state.wfc_ns.adjacency_directions_rc.items():
        wfc_state.compatible_count[coords_index.row, coords_index.column, pattern_id, direction_id] = 0
    wfc_state.observation_stack.append(BanEntry(coords_index.row, coords_index.column, pattern_id))
    wfc_state = RecalculateWeights(wfc_state, coords_index, pattern_id)

    if wfc_state.overflow_check:
        if np.any(np.count_nonzero(wfc_state.wave_table, axis=2) > 60000):
            print("overflow F")
            #print(wfc_state.sums_of_ones)
            print('---')
            print(np.count_nonzero(wfc_state.wave_table, axis=2))
            pdb.set_trace()
            assert False

    
    wfc_state.touch_time[coords_index.row, coords_index.column] = wfc_state.current_iteration_count_last_touch
    if 1 == np.count_nonzero(wfc_state.wave_table[coords_index.row, coords_index.column]):
        if 0 == wfc_state.propagation_time[coords_index.row, coords_index.column]:
            wfc_state.touch_time[coords_index.row, coords_index.column] = wfc_state.current_iteration_count_last_touch
    wfc_state.propagation_time[coords_index.row, coords_index.column] = wfc_state.current_iteration_count_propagation

    if wfc_state.overflow_check:
        if np.any(np.count_nonzero(wfc_state.wave_table, axis=2) > 60000):
            print("overflow G")
            print(np.count_nonzero(wfc_state.wave_table, axis=2))
            pdb.set_trace()
            assert False

    
    #if WFC_FAILURE == wfc_state.result:
    #    return wfc_state
    if 1 == np.count_nonzero(wfc_state.wave_table[coords_index.row, coords_index.column]):
        pattern_flags = wfc_state.wave_table[coords_index.row, coords_index.column]
        wfc_state.output_grid[coords_index.row, coords_index.column] = find_first(True, pattern_flags) # Update the output grid as we go...
        wfc_state.crystal_time[coords_index.row, coords_index.column] = wfc_state.current_iteration_count_crystal
        wfc_state.current_iteration_count_crystal += 1
        if 0 == wfc_state.method_time[coords_index.row, coords_index.column]:
            wfc_state.method_time[coords_index.row, coords_index.column] = wfc_state.current_iteration_count_crystal #(coords_index.row * coords_index.column) + 

    #if WFC_DEBUGGING:
    #   status_print_helper(f"wfc_state.entropies :  {wfc_state.entropies}")
    #    status_print_helper(f"wfc_state.sums_of_ones: {wfc_state.sums_of_ones}")
    ##print(f"Ban({coords_index}, {pattern_id})")
    ##print_internals(wfc_state)

    if wfc_state.overflow_check:
        if np.any(np.count_nonzero(wfc_state.wave_table, axis=2) > 60000):
            print("overflow D")
            print(np.count_nonzero(wfc_state.wave_table, axis=2))
            pdb.set_trace()
            assert False
   
    return wfc_state


# In[35]:


def print_internals(wfc_state):
    show_rendered_patterns(wfc_state, partial=True)
    y = wfc_state.wave_table.shape[0]
    x = wfc_state.wave_table.shape[1]
    print("sums_of_ones")
    for i in range(y):
        for j in range(x):
            print("{0: >02d}".format(wfc_state.wave_table[i,j].sum()), end=' ')
        print()
    print("observation_stack")
    for i in wfc_state.observation_stack:
        print(i.coords_row, i.coords_column, i.pattern_id)
    print("output_grid")
    for i in range(y):
        for j in range(x):
            if wfc_state.wave_table[i,j].sum() > 1:
                print("**", end=' ')
            else:
                try:
                    if len(wfc_state.partial_output_grid.shape) > 2:
                        for k in range(wfc_state.partial_output_grid.shape[2]):
                            if wfc_state.partial_output_grid[i,j,k] != -9:
                                print("{0: >02d}".format(wfc_state.partial_output_grid[i,j,k]), end='+')
                        print(" ", end='')
                    else:
                        print("{0: >02d}".format(wfc_state.partial_output_grid[i,j]), end=' ')
                except:
                    print("??", end=' ')
        print()


# In[36]:

import pdb


def find_upper_left_entropy(wfc_state, random_variation):
    print(wfc_state.wave_table)
    print(np.count_nonzero(wfc_state.wave_table, axis=2))
    print(np.argmax(np.count_nonzero(wfc_state.wave_table, axis=2)))
    pdb.set_trace()
    chosen_cell = np.argmax(np.count_nonzero(wfc_state.wave_table, axis=2))
    if np.all(1 == np.count_nonzero(wfc_state.wave_table, axis=2)):
        status_print_helper("FINISHED")
        if WFC_DEBUGGING:
            status_print_helper(wfc_state.wave_table)
        return WFC_FINISHED
    cell_index = np.unravel_index(chosen_cell, [wfc_state.wave_table.shape[0], wfc_state.wave_table[1]])
    return CoordRC(row=cell_index[0], column=cell_index[1])


def find_upper_left_unresolved(wfc_state, random_variation):
    unresolved_cells = (np.count_nonzero(wfc_state.wave_table, axis=2) > 1)
    unresolved_indices = np.where(unresolved_cells)
    cell_index = (unresolved_indices[0][0], unresolved_indices[1][0])
    return CoordRC(row=cell_index[0], column=cell_index[1])

def find_random_unresolved(wfc_state, random_variation):
    global temp_track_number_of_finishes
    # if np.any(np.count_nonzero(wfc_state.wave_table, axis=2) == 0):
    #     print("FAIL")
    #     return WFC_FAILURE

    # the_result = check_completion(wfc_state)
    # if 0 != the_result:
    #     return the_result

    
    noise_level = 1e-6
    entropy_map = (random_variation * noise_level)
    unresolved_cells = (np.count_nonzero(wfc_state.wave_table, axis=2) > 1)
    entropy_map = entropy_map.flatten() * (0 == unresolved_cells.flatten())
    
    chosen_cell = np.argmax(entropy_map)
    
    cell_index = np.unravel_index(chosen_cell, [np.count_nonzero(wfc_state.wave_table, axis=2).shape[0], np.count_nonzero(wfc_state.wave_table, axis=2).shape[1]])
    return CoordRC(row=cell_index[0], column=cell_index[1])

temp_track_number_of_finishes = 0

def check_completion(wfc_state):
    if np.all(1 == np.count_nonzero(wfc_state.wave_table, axis=2)):
        # Require that every pattern be use at least once
        pattern_set = set(np.argmax(wfc_state.wave_table, axis=2).flatten())
        # Force a test to encourage backtracking - temporary addition
        if (len(pattern_set) != wfc_state.number_of_patterns) and wfc_state.wfc_ns.force_use_all_patterns:
            print("Some patterns were not used")
            return WFC_FAILURE

        # Force a test to encourage backtracking - temporary addition
        if (not (57 in pattern_set)) and False:
            print("Ground not used")
            return WFC_FAILURE

        #Just force a failure the first time for testing purposes
        temp_track_number_of_finishes += 1
        if temp_track_number_of_finishes < 2 and wfc_state.wfc_ns.force_fail_first_solution:
            print("Force fake failure to test backtracking")
            return WFC_FAILURE

        
        status_print_helper("FINISHED")
        if WFC_DEBUGGING:
            status_print_helper(wfc_state.wave_table)
        return WFC_FINISHED
    if (np.any(np.count_nonzero(wfc_state.wave_table, axis=2) < 1)):
        return WFC_FAILURE
    return None

def find_upper_left_relevant(wave_table, random_variation):
    unresolved_cells = (np.count_nonzero(wave_table, axis=-1) > 1)
    rows, cols = np.where(unresolved_cells)
    return CoordRC(row=rows[0], column=cols[0])

def find_random_unresolved_relevant(wave_table, random_variation):
    unresolved_cell_mask = (np.count_nonzero(wave_table, axis=2) > 1)
    cell_weights = np.where(unresolved_cell_mask, random_variation, np.inf)
    row, col = np.unravel_index(np.argmin(cell_weights), cell_weights.shape)
    return CoordRC(row=row, column=col)

def find_minimum_entropy_relevant(wave_table, random_variation):
    unresolved_cell_mask = (np.count_nonzero(wave_table, axis=2) > 1)
    cell_weights = np.where(unresolved_cell_mask, random_variation + (np.count_nonzero(wave_table, axis=2)),  np.inf)
    row, col = np.unravel_index(np.argmin(cell_weights), cell_weights.shape)
    return CoordRC(row=row, column=col)

def find_minimum_entropy(wfc_state, random_variation):
    global temp_track_number_of_finishes
    noise_level = 1e-6
    entropy_map = (random_variation * noise_level)
    entropy_map = entropy_map.flatten() + wfc_state.entropies.flatten()
    # TODO: add boundary check for non-wrapping generation


    minimum_cell = np.argmin(entropy_map)
    if np.any(np.count_nonzero(wfc_state.wave_table, axis=2) == 0):
        print("FAIL")
        #print(np.count_nonzero(wfc_state.wave_table, axis=2))
        print(f"previous decisions: {len(wfc_state.previous_decisions)}")

        return WFC_FAILURE
    if 0 == np.count_nonzero(wfc_state.wave_table, axis=2).flatten()[minimum_cell]:
        if WFC_DEBUGGING:
            print_internals(wfc_state)
        return WFC_FAILURE

    
    higher_than_threshold = np.ma.MaskedArray(entropy_map, np.count_nonzero(wfc_state.wave_table, axis=2).flatten() <= 1)
    minimum_cell = higher_than_threshold.argmin(fill_value=999999.9)#np.ma.maximum_fill_value(1))
    maximum_cell = higher_than_threshold.argmax(fill_value=0.0)
    chosen_cell = maximum_cell

    if np.any(np.count_nonzero(wfc_state.wave_table, axis=2) == 0):
        wfc_logger.debug("A zero-state node has been found.")


    if wfc_state.overflow_check:
        if(np.any(np.count_nonzero(wfc_state.wave_table, axis=2) > 65534)):
            wfc_logger.error("overflow A")
            wfc_logger.error(np.count_nonzero(wfc_state.wave_table, axis=2))
            pdb.set_trace()
            assert False

        
    if np.all(1 == np.count_nonzero(wfc_state.wave_table, axis=2)):
        print("DETECTED FINISH")
        print("nonzero count: {np.count_nonzero(wfc_state.wave_table, axis=2)}")

    #the_result = check_completion(wfc_state)
    #if 0 != the_result:
    #    return the_result
    
    
    cell_index = np.unravel_index(chosen_cell, [np.count_nonzero(wfc_state.wave_table, axis=2).shape[0], np.count_nonzero(wfc_state.wave_table, axis=2).shape[1]])
    return CoordRC(row=cell_index[0], column=cell_index[1])

def FinalizeObservedWaves(wfc_state):
    for row in range(wfc_state.rows): 
        for column in range(wfc_state.columns):
            pattern_flags = wfc_state.wave_table[row, column]
            wfc_state.output_grid[row, column] = find_first(True, pattern_flags) # TODO: This line is probably overkill?
    wfc_state.result = WFC_FINISHED
    return wfc_state

def make_observation_relevant(state, parameters, instrumentation, dirty_cell_list, cell_to_observe, random_number_generator):
    row, column = cell_to_observe
    distribution = np.zeros((parameters.number_of_patterns,), dtype=np.float64)
    for wave_pat in range(parameters.number_of_patterns):
        if state.wave_table[row, column, wave_pat]:
            distribution[wave_pat] = parameters.weights[wave_pat]

    cell_weight_sum = sum(distribution)
    if cell_weight_sum <= 0:
        wfc_logger.info(f"Tried to observe cell with no valid weights: {cell_to_observe} is {cell_weight_sum}")
        return state, instrumentation, dirty_cell_list
    normalized = [float(i) / cell_weight_sum for i in distribution]

    choice_count = sum([1 for i in distribution if i > 0])
    chosen_pattern = parameters.patterns[0]
    chosen_pattern = random_number_generator.choice(parameters.patterns, 1, p=normalized)[0]
    instrumentation.choices_recording[row, column] = math.log(choice_count)

    if parameters.visualizing_output:
        instrumentation.output_grid[row, column] = chosen_pattern
        instrumentation.solving_time[row, column] = wfc_state.current_iteration_count_observation
        instrumentation.touch_time[row, column] = wfc_state.current_iteration_count_last_touch

    for wave_pat in range(parameters.number_of_patterns):
        if state.wave_table[row, column][wave_pat] != (wave_pat == chosen_pattern):
            state.wave_table[cell_to_observe.row, cell_to_observe.column, chosen_pattern] = False
            for direction_id, direction_offset in parameters.wfc_ns.adjacency_directions_rc.items():
                state.compatible_count[row, column, chosen_pattern, direction_id] = 0
                state.observation_stack.append(BanEntry(row, column, chosen_pattern))
                state = recalculate_weights(state, parameters, cell_to_observe, chosen_pattern)
   
    return state, instrumentation, dirty_cell_list

def make_observation(wfc_state, cell_to_observe, random_number_generator):
    print(dir(wfc_state))
    assert False
    row, column = cell_to_observe
    distribution = np.zeros((wfc_state.number_of_patterns,), dtype=np.float64)
    for wave_pat in range(wfc_state.number_of_patterns):
        if wfc_state.wave_table[row, column, wave_pat]:
            distribution[wave_pat] = wfc_state.weights[wave_pat]
    
    cell_weight_sum = sum(distribution)
    normalized = [float(i) / cell_weight_sum for i in distribution]
    if (np.any(np.isnan(normalized))):
        print(normalized)
        print(distribution)
        print(cell_weight_sum)
        
    #assert not(np.any(np.isnan(normalized)))
        
    choice_count = sum([1 for i in distribution if i > 0])
    chosen_pattern = wfc_state.patterns[0]
    try:
        chosen_pattern = random_number_generator.choice(wfc_state.patterns, 1, p=normalized)[0]
        wfc_state.choices_recording[row, column] = math.log(choice_count)
    except ValueError as e:
        print("observation ValueError")
        print(e)
        print(normalized)
    if WFC_DEBUGGING:
        print("chosen_pattern: {0}".format(chosen_pattern))
        print("wfc_state.patterns[chosen_pattern]: {0}".format(wfc_state.patterns[chosen_pattern]))

    if wfc_state.visualizing_output:
        wfc_state.output_grid[row, column] = chosen_pattern
        wfc_state.solving_time[row, column] = wfc_state.current_iteration_count_observation
        wfc_state.touch_time[row, column] = wfc_state.current_iteration_count_last_touch
        wfc_state.method_time[row, column] = 1000 + wfc_state.current_iteration_count_observation

    
    #wave = wfc_state.wave_table[row, column]
    for wave_pat in range(wfc_state.number_of_patterns):
        if wfc_state.wave_table[row, column][wave_pat] != (wave_pat == chosen_pattern):
            #wfc_state = Ban(wfc_state, cell_to_observe, wave_pat)
            wfc_state.wave_table[cell_to_observe.row, cell_to_observe.column, pattern_id] = False
            for direction_id, direction_offset in wfc_state.wfc_ns.adjacency_directions_rc.items():
                wfc_state.compatible_count[coords_index.row, coords_index.column, pattern_id, direction_id] = 0
                wfc_state.observation_stack.append(BanEntry(coords_index.row, coords_index.column, pattern_id))
                wfc_state = RecalculateWeights(wfc_state, coords_index, pattern_id)
                
                

            
    wfc_state.wfc_ns.stats_tracking["observations"] += 1
    global ongoing_observations
    ongoing_observations += 1
    wfc_state.wfc_ns.stats_tracking["total_observations"] = ongoing_observations
    if wfc_state.wfc_ns.backtracking:
        wfc_state.previous_decisions.append((cell_to_observe, wave_pat,))

    if wfc_state.logging:
        with open(wfc_state.wfc_ns.debug_log_filename, "a") as stats_file:
            stats_file.write(f"making observation at: {cell_to_observe}\n")
            stats_file.write(f"{wave_pat}\n")

    
    return wfc_state

def wfc_observe(wfc_state, random_variation, random_number_generator):
    wfc_state.current_iteration_count_observation += 1

    the_result = None
    if np.all(1 == np.count_nonzero(wfc_state.wave_table, axis=2)):
        status_print_helper("FINISHED")
        if WFC_DEBUGGING:
            status_print_helper(wfc_state.wave_table)
        the_result = WFC_FINISHED
    if None == the_result:
        the_result = check_completion(wfc_state)

    cell = None
    if None == the_result:
        cell = find_minimum_entropy(wfc_state, random_variation)
        #cell = find_upper_left_entropy(wfc_state, random_variation)
        #cell = find_upper_left_unresolved(wfc_state, random_variation)
        #cell = find_random_unresolved(wfc_state, random_variation)

    if cell == WFC_FAILURE:
        the_result = cell

    if np.all(1 == np.count_nonzero(wfc_state.wave_table, axis=2)):
        status_print_helper("FINISHED")
        if WFC_DEBUGGING:
            status_print_helper(wfc_state.wave_table)
        the_result = WFC_FINISHED
    if None == the_result:
        the_result = check_completion(wfc_state)
        
    #print(f"&&& We are observing cell: {cell}")
    #if the_result != 0:
    #    print(f"result: {the_result}")
    if WFC_FAKE_FAILURE == the_result:
        wfc_state.result = WFC_FAILURE
        wfc_state.fake_failure = True
        return wfc_state
    if WFC_FAILURE == the_result:
        wfc_state.result = WFC_FAILURE
        return wfc_state
    if WFC_FINISHED == the_result:
        return FinalizeObservedWaves(wfc_state)

    return make_observation(wfc_state, cell, random_number_generator)


# In[37]:


def show_crystal_time(wfc_state, partial=False):
    #wfc_state.solving_time
    #wfc_state.propagation_time
    #pl = matshow(wfc_state.solving_time, cmap='gist_ncar', extent=(0, wfc_state.rows, wfc_state.columns, 0))
    #pl.axes.grid(None)
    #pl = matshow(wfc_state.propagation_time, cmap='gist_ncar', extent=(0, wfc_state.rows, wfc_state.columns, 0))
    #pl.axes.grid(None)
    
    matfig_obsv = figure(figsize=(9,9))

    ax = subplot(1,1,1)
    title('Observation Time')
    ax.matshow(wfc_state.solving_time,cmap='viridis')
    ax.grid(None)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(None)
    #matfig_obsv.colorbar()
    print(f"solving time: {wfc_state.solving_time}")
    
    matfig_prop = figure(figsize=(9,9))
    ax = subplot(1,1,1)
    title('Propagation Time')
    ax.matshow(wfc_state.propagation_time,cmap='viridis')
    ax.grid(None)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(None)
    #matfig_prop.colorbar()
    print(f"propagation time: {wfc_state.propagation_time}")

    matfig_touch = figure(figsize=(9,9))
    ax = subplot(1,1,1)
    title('Last Altered Time')
    ax.matshow(wfc_state.touch_time,cmap='viridis')
    ax.grid(None)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(None)
    #matfig_prop.colorbar()
    print(f"touch time: {wfc_state.touch_time}")

    matfig_touch2 = figure(figsize=(9,9))
    ax = subplot(1,1,1)
    title('Crystal Time')
    ax.matshow(wfc_state.crystal_time,cmap='viridis')
    ax.grid(None)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(None)
    #matfig_prop.colorbar()
    print(f"touch_time: {wfc_state.touch_time}")
    matplotlib.pyplot.close(matfig_obsv)
    matplotlib.pyplot.close(matfig_prop)
    matplotlib.pyplot.close(matfig_touch)
    matplotlib.pyplot.close(matfig_touch2)

def show_rendered_patterns(wfc_state, partial=False):
    partial=True
    pattern_grid = np.array(wfc_state.output_grid, dtype=np.int64)
    preview_size = (wfc_state.wfc_ns.pattern_width * wfc_state.rows, wfc_state.wfc_ns.pattern_width * wfc_state.columns)
    has_gaps = np.any(np.count_nonzero(wfc_state.wave_table, axis=2) != 1)
    #print(wfc_state.sums_of_ones != 1)
    #print(f"has gaps: {has_gaps}", end=', ')
    print(f"remaining nodes: {np.count_nonzero(np.count_nonzero(wfc_state.wave_table, axis=2) != 1)}")

    # Temporarily only look at the final few node resulrt...
    #if np.count_nonzero(np.count_nonzero(wfc_state.wave_table, axis=2) != 1) > 10:
    #    return
    
    if has_gaps:
        pattern_grid = np.array(wfc_state.partial_output_grid, dtype=np.int64)
        preview_size = (wfc_state.wfc_ns.pattern_width * wfc_state.rows, wfc_state.wfc_ns.pattern_width * wfc_state.columns, wfc_state.partial_output_grid.shape[2])
    grid_preview = np.full(preview_size, WFC_PARTIAL_BLANK, dtype=np.int64)
    #pattern_center = (math.floor((wfc_state.wfc_ns.pattern_width - 1) / 2), math.floor((wfc_state.wfc_ns.pattern_width - 1) / 2))
    pattern_center = wfc_state.wfc_ns.pattern_center
    
    special_count_tiles = np.full((wfc_state.wfc_ns.pattern_width,wfc_state.wfc_ns.pattern_width), 1, dtype=np.int64)
    #WFC_DEBUGGING = True
    #print(len(pattern_grid.shape))
    if WFC_DEBUGGING:
        print("show rendered patterns")
    for row in range(wfc_state.rows):
        #print()
        if WFC_DEBUGGING:
            print()
        for column in range(wfc_state.columns):
            if (len(pattern_grid.shape) > 2):
                if WFC_DEBUGGING: 
                    print('[',end='')
                pattern_list = []
                for z in range(wfc_state.number_of_patterns):
                    pattern_list.append(pattern_grid[(row,column,z)])
                pl_count = 0
                for the_pattern in pattern_list:
                    if WFC_DEBUGGING:
                        print(the_pattern, end='')
                    if (the_pattern != -1) and (the_pattern != WFC_NULL_VALUE):
                        if WFC_DEBUGGING:
                            print('!', end='')
                        the_pattern_tiles = wfc_state.wfc_ns.pattern_catalog[the_pattern]
                        grid_preview = blit(grid_preview, the_pattern_tiles, (row*wfc_state.wfc_ns.pattern_width,column*wfc_state.wfc_ns.pattern_width), layer = pl_count)
                        pl_count += 1
                    else:
                        if WFC_DEBUGGING:
                            print(' ', end='')
                    if WFC_DEBUGGING:
                        print('',end=' ')
                if WFC_DEBUGGING:
                    print(']',end='')
            else:
                if WFC_DEBUGGING:
                    print(pattern_grid)
                #print(pattern_grid[(row,column)],end=' ')
                if WFC_NULL_VALUE != pattern_grid[(row,column)]:
                    the_pattern = wfc_state.wfc_ns.pattern_catalog[pattern_grid[(row,column)]]
                    #if not partial:
                    #    p_x = wfc_state.wfc_ns.pattern_center[0]
                    #    p_y = wfc_state.wfc_ns.pattern_center[1]
                    #    the_pattern = the_pattern[p_x:p_x+1, p_y:p_y+1]
                    #print(f"the_pattern: {the_pattern}")
                    grid_preview = blit(grid_preview, the_pattern, (row * wfc_state.wfc_ns.pattern_width, column * wfc_state.wfc_ns.pattern_width))
    if WFC_DEBUGGING:
        print(f"\ngrid_preview:\n{grid_preview}")
    ptr = tiles_to_images(wfc_state.wfc_ns, grid_preview, wfc_state.wfc_ns.tile_catalog, wfc_state.wfc_ns.tile_size, visualize=True, partial=partial).astype(np.uint8)
    if WFC_DEBUGGING:
        print(f"ptr: {ptr}")
    fig, ax = subplots(figsize=(16,16))
    ax.grid(color="magenta", linewidth=1.5)
    ax.tick_params(direction='in', bottom=False, left=False)

    im = ax.imshow(ptr)
    for axis, dim in zip([ax.xaxis, ax.yaxis],[wfc_state.columns, wfc_state.rows]):
        axis.set_ticks(np.arange(-0.5, (wfc_state.wfc_ns.pattern_width * dim) + 0.5, wfc_state.wfc_ns.pattern_width))
        axis.set_ticklabels([])
        


# In[38]:


def render_patterns_to_output(wfc_state, partial=False, visualize=True):
    pattern_grid = np.array(wfc_state.output_grid, dtype=np.int64)
    
    has_gaps = np.any(np.count_nonzero(wfc_state.wave_table, axis=2) != 1) 
    if has_gaps:
        pattern_grid = np.array(wfc_state.partial_output_grid, dtype=np.int64)
    render_grid = np.full(pattern_grid.shape,  WFC_PARTIAL_BLANK, dtype=np.int64)
    pattern_center = wfc_state.wfc_ns.pattern_center
    for row in range(wfc_state.rows):
        if WFC_DEBUGGING:
            print()
        for column in range(wfc_state.columns):
            if (len(pattern_grid.shape) > 2):
                if WFC_DEBUGGING:
                    print('[',end='')
                pattern_list = []
                for z in range(wfc_state.number_of_patterns):
                    pattern_list.append(pattern_grid[(row,column,z)])
                pattern_list = [pattern_grid[(row,column,z)] for z in range(wfc_state.number_of_patterns) if (pattern_grid[(row,column,z)] != -1) and (pattern_grid[(row,column,z)] != WFC_NULL_VALUE)]
                for pl_count, the_pattern in enumerate(pattern_list):
                    if WFC_DEBUGGING:
                        print(the_pattern, end='')
                    the_pattern_tiles = wfc_state.wfc_ns.pattern_catalog[the_pattern][pattern_center[0]:pattern_center[0]+1,pattern_center[1]:pattern_center[1]+1]
                    if WFC_DEBUGGING:
                        print(the_pattern_tiles, end=' ')
                    render_grid = blit(render_grid, the_pattern_tiles, (row,column), layer = pl_count)
                if WFC_DEBUGGING:
                    print(']',end=' ')
            else:
                if WFC_DEBUGGING:
                    print(pattern_grid[(row,column)], end=',')
                if WFC_NULL_VALUE != pattern_grid[(row,column)]:
                    the_pattern = wfc_state.wfc_ns.pattern_catalog[pattern_grid[(row,column)]]
                    p_x = wfc_state.wfc_ns.pattern_center[0]
                    p_y = wfc_state.wfc_ns.pattern_center[1]
                    the_pattern = the_pattern[p_x:p_x+1, p_y:p_y+1]
                    render_grid = blit(render_grid, the_pattern, (row, column))
    if WFC_DEBUGGING:
        print("\nrender grid")
        print(render_grid)
    ptr = tiles_to_images(wfc_state.wfc_ns, render_grid, wfc_state.wfc_ns.tile_catalog, wfc_state.wfc_ns.tile_size, visualize=True, partial=partial).astype(np.uint8)
    if WFC_DEBUGGING:
        print(f"ptr {ptr}")
        
    if visualize:
        fig, ax = subplots(figsize=(16,16))
        #ax.grid(color="magenta", linewidth=1.5)
        ax.tick_params(direction='in', bottom=False, left=False)

        im = ax.imshow(ptr)
        for axis, dim in zip([ax.xaxis, ax.yaxis],[wfc_state.columns, wfc_state.rows]):
            axis.set_ticks(np.arange(-0.5, dim + 0.5, 1))
            axis.set_ticklabels([])
    #print(ptr)
    imageio.imwrite(wfc_state.wfc_ns.output_filename, ptr)




# In[41]:


def is_cell_on_boundary(wfc_state, wfc_coords):
    if not wfc_state.wfc_ns.periodic_output:
        return False
    # otherwise...
    return False # TODO


def wrap_coords(wfc_state, cell_coords):
    r = (cell_coords.row + wfc_state.wfc_ns.generated_size[0]) % (wfc_state.wfc_ns.generated_size[0])
    c = (cell_coords.column + wfc_state.wfc_ns.generated_size[1]) % (wfc_state.wfc_ns.generated_size[1])
    return CoordRC(row=r, column=c)

def wfc_propagate(wfc_state):
    
    # while len(wfc_state.observation_stack) > 0:
    #     element = wfc_state.observation_stack.pop()
    #     print(f"element: {element}")
    #     for direction_id, direction_offset in wfc_state.wfc_ns.adjacency_directions.items():
    #         neighbor_coords = CoordRC(row=element.coords_row + direction_offset.y, column=element.coords_column + direction_offset.x)
    #         if not is_cell_on_boundary(wfc_state, neighbor_coords):
    #             neighbor_coords = wrap_coords(wfc_state, neighbor_coords)
    #             compatible_pattern_list = wfc_state.propagator_matrix[direction_id][element.pattern_id]
    #             for pat_id, pat_value in enumerate(compatible_pattern_list):
    #                 if pat_value:
    #                     wfc_state.compatible_count[neighbor_coords[0], neighbor_coords[1], pat_id, direction_id] -= 1
    #                     if 0 == wfc_state.compatible_count[neighbor_coords[0], neighbor_coords[1], pat_id, direction_id]:
    #                         wfc_state = Ban(wfc_state, neighbor_coords, pat_id)
    # return wfc_state

    #wfc_state = wfc_observe(wfc_state, random_variation, random_number_generator)
    #wfc_state = wfc_partial_output(wfc_state)
    #show_rendered_patterns(wfc_state, True)
    #render_patterns_to_output(wfc_state, True)

    #print(f"Propagating. Current solver result: {wfc_state.result}")
    assert(wfc_state.result == None)
    #print(wfc_state.compatible_count.shape)
    while(len(wfc_state.observation_stack) > 0):
        #print(wfc_state.observation_stack)
        if wfc_state.overflow_check:
            if np.any(np.count_nonzero(wfc_state.wave_table, axis=2) > 60000):
                print("overflow K")
                print(np.count_nonzero(wfc_state.wave_table, axis=2))
                assert False

        
        wfc_state.current_iteration_count_propagation += 1

        element = wfc_state.observation_stack.pop()
        #print(f"element: {element}")
        for direction_id, direction_offset in wfc_state.wfc_ns.adjacency_directions.items():
            neighbor_coords = CoordRC(row=element.coords_row + direction_offset.y, column=element.coords_column + direction_offset.x)
            neighbor_coords = wrap_coords(wfc_state, neighbor_coords)
            #print(f" {direction_offset} -> {neighbor_coords}")
            compatible_pattern_list = wfc_state.propagator_matrix[direction_id, element.pattern_id]
            #print(compatible_pattern_list)
            for pat_id, pat_value in enumerate(compatible_pattern_list):
                if pat_value:
                    #print(neighbor_coords)
                    wfc_state.compatible_count[neighbor_coords.row, neighbor_coords.column, pat_id, direction_id] -= 1
                    if 0 == wfc_state.compatible_count[neighbor_coords.row, neighbor_coords.column, pat_id, direction_id]:
                        wfc_state = Ban(wfc_state, neighbor_coords, pat_id)

    #print(f"*** stack length: {len(wfc_state.observation_stack)}")
    #print(f"~~~ compatible count:\n{wfc_state.compatible_count}")
    #print(f"~~~ wave table:\n{wfc_state.wave_table}")

    #wfc_state = wfc_partial_output(wfc_state)
    #show_rendered_patterns(wfc_state, True)                
    #render_patterns_to_output(wfc_state, True)
    wfc_state.wfc_ns.stats_tracking["propagations"] += 1
    return wfc_state

import pdb

backtrack_track_global = 0
def wfc_backtrack(current_wfc_state, list_of_old_wfc_states):
    global backtrack_track_global
    backtrack_track_global += 1
    print(f"backtrack {backtrack_track_global}")
    #current_wfc_state.wave_table = copy.deepcopy(old_wfc_state.wave_table)
    #current_wfc_state.compatible_count = copy.deepcopy(old_wfc_state.compatible_count)
    #current_wfc_state.sums_of_ones = copy.deepcopy(old_wfc_state.sums_of_ones)
    #current_wfc_state.sums_of_weights = copy.deepcopy(old_wfc_state.sums_of_weights)
    #current_wfc_state.sums_of_weight_log_weights = copy.deepcopy(old_wfc_state.sums_of_weight_log_weights)
    #current_wfc_state.entropies = copy.deepcopy(old_wfc_state.entropies)
    #current_wfc_state.observation_stack = copy.deepcopy(old_wfc_state.observation_stack)
    before_len = len(current_wfc_state.previous_decisions)
    forbidden = current_wfc_state.previous_decisions.pop()
    assert (before_len != len(current_wfc_state.previous_decisions))
    wfc_logger.warning(f"Backtracking from {forbidden}")

    if current_wfc_state.logging:
        with open(current_wfc_state.wfc_ns.debug_log_filename, "a") as stats_file:
            stats_file.write(f"Backtracking #{backtrack_track_global}\n")
            stats_file.write("Past state list:\n")
            past_state_list = list(zip([np.count_nonzero(np.count_nonzero(o_wfc_state.wave_table, axis=2) > 1) for o_wfc_state in list_of_old_wfc_states], [np.count_nonzero(np.count_nonzero(o_wfc_state.wave_table, axis=2) < 1) for o_wfc_state in list_of_old_wfc_states]))
            stats_file.write(f"{past_state_list}\n")

            


    
    #print("Past state list: ", end='')
    #print([np.count_nonzero(np.count_nonzero(o_wfc_state.wave_table, axis=2) > 1) for o_wfc_state in list_of_old_wfc_states], end=' | ')
    #print([np.count_nonzero(np.count_nonzero(o_wfc_state.wave_table, axis=2) < 1) for o_wfc_state in list_of_old_wfc_states])
    old_wfc_state = list_of_old_wfc_states.pop()
    #print(np.count_nonzero(old_wfc_state.wave_table, axis=2))
    #print(f"remaining nodes: {np.count_nonzero(np.count_nonzero(old_wfc_state.wave_table, axis=2) != 1)}")
    #print(f"invalid nodes: {(np.count_nonzero(np.count_nonzero(old_wfc_state.wave_table, axis=2) < 1)) + (np.count_nonzero(np.count_nonzero(old_wfc_state.wave_table, axis=2) > 60000))}")

    #print("vvvvv")
    try:
        old_wfc_state = list_of_old_wfc_states.pop()
    except IndexError as e:
        wfc_logger.info("stack of previous states is empty")
        if old_wfc_state.logging:
            with open(old_wfc_state.wfc_ns.debug_log_filename, "a") as stats_file:
                stats_file.write(f"stack of previous states is empty\n")
    #print(np.count_nonzero(old_wfc_state.wave_table, axis=2))
    #print(f"remaining nodes: {np.count_nonzero(np.count_nonzero(old_wfc_state.wave_table, axis=2) != 1)}")
    #print(f"invalid nodes: {(np.count_nonzero(np.count_nonzero(old_wfc_state.wave_table, axis=2) < 1)) + (np.count_nonzero(np.count_nonzero(old_wfc_state.wave_table, axis=2) > 60000))}")

    #print("Past state list: ", end='')
    #print([np.count_nonzero(np.count_nonzero(o_wfc_state.wave_table, axis=2) > 1) for o_wfc_state in list_of_old_wfc_states], end = ' ')
    #print([np.count_nonzero(np.count_nonzero(o_wfc_state.wave_table, axis=2) < 1) for o_wfc_state in list_of_old_wfc_states])

    
    nold_wfc_state = copy.deepcopy(old_wfc_state)
    #wfc_logger.warning(f"nold ones: {nold_wfc_state.sums_of_ones}")

    #current_wfc_state = Ban(current_wfc_state, forbidden[0], forbidden[1])
    #current_wfc_state.result = None
    new_wfc_state = Ban(nold_wfc_state, forbidden[0], forbidden[1])
    #print(f"new invalid nodes: {(np.count_nonzero(np.count_nonzero(new_wfc_state.wave_table, axis=2) < 1)) + (np.count_nonzero(np.count_nonzero(new_wfc_state.wave_table, axis=2) > 60000))}")
    #wfc_logger.warning(f"new ones: {new_wfc_state.sums_of_ones}")
    if new_wfc_state.overflow_check:
        if np.any(np.count_nonzero(new_wfc_state.wave_table, axis=2) > 60000):
            print("overflow J")
            print(np.count_nonzero(new_wfc_state.wave_table, axis=2))
            assert False
    #wfc_logger.warning(f"obstack len {len(new_wfc_state.observation_stack)}")
    #new_wfc_state.observation_stack.pop()
    #wfc_logger.warning(f"obstack len {len(new_wfc_state.observation_stack)}")

    #pdb.set_trace()

    new_wfc_state.result = None

    if new_wfc_state.logging:
        with open(new_wfc_state.wfc_ns.debug_log_filename, "a") as stats_file:
            stats_file.write(f"After backtracking:\n")
            stats_file.write(f"stack length:{len(list_of_old_wfc_states)}\n")
            stats_file.write("remaining wave table choices:\n")
            stats_file.write(f"{(np.count_nonzero(new_wfc_state.wave_table, axis=2))}\n")
    
    return new_wfc_state, list_of_old_wfc_states

#from lucid_serialize_array import _normalize_array

import cProfile, pstats
import logging

ongoing_observations = 0

def reset_backtracking_count():
    global backtrack_track_global
    backtrack_track_global = 0

def wfc_run(wfc_seed_state, visualize=False, logging=False):
    wfc_logger.info("wfc_run()")
    global WFC_VISUALIZE
    if visualize:
        WFC_VISUALIZE = True
    else:
        WFC_VISUALIZE = False

    global ongoing_observations
    ongoing_observations = 0

    
    wfc_state = wfc_init(wfc_seed_state)  
    
    #print("Profiling clear...")  
    #pr = cProfile.Profile()
    #pr.enable()   
    wfc_state = wfc_clear(wfc_state)
    #pr.disable()
    #print("Profiling complete...")
    #profile_filename = "" + str(wfc_state.wfc_ns.output_path) + "" + "clear_" + str(wfc_state.wfc_ns.output_file_number) + "_" + str(wfc_state.wfc_ns.seed) + "_" + str(time.time()) + ".profile"
    #with open(profile_filename, 'w') as profile_file:
    #    ps = pstats.Stats(pr, stream=profile_file)
    #    ps.print_stats()
    #print("...profile saved")
        
    wfc_state.logging = logging
    wfc_state.overflow_check = False
    
    if visualize:
        show_pattern_adjacency(wfc_state)
        visualize_propagator_matrix(wfc_state.propagator_matrix)
    random_number_generator = np.random.RandomState(wfc_state.wfc_ns.seed)
    random_variation = random_number_generator.random_sample(wfc_state.entropies.size)

    recorded_vis = None
    if visualize:
        recorded_vis = record_visualization(wfc_state, recorded_vis)
    vis_stack = []
    
    backtracking_stack = []
    backtracking_count = 0
    print(wfc_state.patterns)
    
    iterations = 0
    while (iterations < wfc_state.wfc_ns.iteration_limit) or (0 == wfc_state.wfc_ns.iteration_limit):
        wfc_state.backtracking_count = backtracking_count
        wfc_state.backtracking_stack_length = len(backtracking_stack)
        wfc_state.backtracking_total = backtrack_track_global
        if visualize:
            recorded_vis = record_visualization(wfc_state, recorded_vis)
        wfc_state.current_iteration_count_last_touch += 1
        wfc_state = wfc_observe(wfc_state, random_variation, random_number_generator)

        # Add a time-out on the number of total observations
        #print(f"Observations so far: {wfc_state.wfc_ns.stats_tracking['total_observations']}")
        if (wfc_state.wfc_ns.stats_tracking["total_observations"] > 3000):
            wfc_state.result = WFC_TIMEDOUT
            return wfc_state
        
        #print(f"wfc_state.entropies :  {wfc_state.entropies}")
        if visualize:
            vis_stack.append(visualize_entropies(wfc_state))
        if iterations % 50 == 0:
            print(iterations, end=' ')#print(np.count_nonzero(wfc_state.wave_table, axis=2))
            #print(np.argmax(wfc_state.wave_table, axis=2))
        #print(wfc_state.result)
        #print_internals(wfc_state)


        if wfc_state.logging:
            with open(wfc_state.wfc_ns.debug_log_filename, "a") as stats_file:
                stats_file.write(f"\n=====\n")
                stats_file.write(f"result: {wfc_state.result}\n")
                stats_file.write(f"total observations: {wfc_state.wfc_ns.stats_tracking['total_observations']}\n")
                stats_file.write(f"On backtracking {backtracking_count}, with stack size {len(backtracking_stack)}\n")
                stats_file.write(f"{wfc_state.result}\n")
                stats_file.write("remaining wave table choices:\n")
                stats_file.write(f"{(np.count_nonzero(wfc_state.wave_table, axis=2))}\n")
            
            
        
        if WFC_FINISHED == wfc_state.result:
            wfc_state.wfc_ns.stats_tracking['success'] = True
            wfc_state.recorded_vis = recorded_vis
            return wfc_state
        if WFC_FAILURE == wfc_state.result:
            #print(np.count_nonzero(wfc_state.wave_table, axis=2))
            if not wfc_state.wfc_ns.backtracking:
                wfc_state.recorded_vis = recorded_vis
                return wfc_state
            if (len(backtracking_stack) <= 0):
                wfc_state.recorded_vis = recorded_vis
                return wfc_state
            if (len(wfc_state.previous_decisions) <= 0):
                wfc_state.recorded_vis = recorded_vis
                return wfc_state
            backtracking_count += 1
            wfc_logger.warning(f"Backtracking {backtracking_count}, stack size {len(backtracking_stack)}")
            if backtracking_count > 450: # Time out on too many backtracks
                wfc_state.result = WFC_TIMEDOUT
                wfc_state.recorded_vis = recorded_vis
                return wfc_state
            
            #for ix, bs in enumerate(backtracking_stack):
            #    print(ix, end=',')
                #print(bs.sums_of_ones)
            #last_backtracking_added = backtracking_stack.pop()
            wfc_state, backtracking_stack = wfc_backtrack(wfc_state, backtracking_stack)
            wfc_logger.warning(f"stack size {len(backtracking_stack)}")
            #print(f"wfc_state.sums_of_ones\n{wfc_state.sums_of_ones}")
            print(f"current result code: {wfc_state.result}")
            #if len(backtracking_stack) <= 0:
            #    #wfc_state.recorded_vis = recorded_vis
            #    #return wfc_state
            #    print(f"backtracking stack empty")
            
        if visualize:
            wfc_state = wfc_partial_output(wfc_state)
            visualize_compatible_count(wfc_state)
            visualize_entropy(wfc_state)
            show_rendered_patterns(wfc_state, True)
        
        wfc_state = wfc_propagate(wfc_state)
        if visualize:
            wfc_state = wfc_partial_output(wfc_state)
        #show_rendered_patterns(wfc_state, partial=True)
        #render_patterns_to_output(wfc_state, partial=True)
        iterations += 1
        #print(iterations, end = ' ')
        #wfc_logger.info("iterations: " + str(iterations))
        backtracking_stack.append(copy.deepcopy(wfc_state))
    wfc_state.result = WFC_TIMEDOUT
    
    wfc_state.recorded_vis = recorded_vis
    return wfc_state

if __name__ == "__main__":
    import doctest
    doctest.testmod()

