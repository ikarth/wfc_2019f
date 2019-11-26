"Visualize the patterns into tiles and so on."

from .wfc_patterns import pattern_grid_to_tiles
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import imageio
import math
import pathlib
import itertools

## Helper functions
RGB_CHANNELS = 3
def rgb_to_int(rgb_in):
  """"Takes RGB triple, returns integer representation."""
  return struct.unpack('I', 
                       struct.pack('<' + 'B' * 4, 
                                   *(rgb_in + [0] * (4 - len(rgb_in)))))[0]
def int_to_rgb(val):
  return [x for x in val.to_bytes(RGB_CHANNELS, 'little')]



def tile_to_image(tile, tile_catalog, tile_size, visualize=False):
    """
    Takes a single tile and returns the pixel image representation.
    """
    new_img = np.zeros((tile_size[0], tile_size[1], 3), dtype=np.int64)
    for u in range(tile_size[0]):
        for v in range(tile_size[1]):
            ## If we want to display a partial pattern, it is helpful to
            ## be able to show empty cells. Therefore, in visualize mode,
            ## we use -1 as a magic number for a non-existant tile.
            pixel = [200, 0, 200]
            if (visualize) and ((-1 == tile) or (WFC_PARTIAL_BLANK == tile)):
                if 0 == (u + v) % 2:
                    pixel = [255, 0, 255]
            else:
                if (visualize) and -2 == tile:
                    pixel = [0, 255, 255]
                else:            
                    pixel = tile_catalog[tile][u,v]
            new_img[u,v] = pixel
    return new_img


#  plt.figure(figsize=(4,4), edgecolor='k', frameon=True)
#  plt.title('Extracted Tiles')
#  s = math.ceil(math.sqrt(len(unique_tiles)))+1
#  for i,tcode in enumerate(unique_tiles[0]):
#    sp = plt.subplot(s, s, i + 1).imshow(tile_catalog[tcode])
#    sp.axes.tick_params(labelleft=False, labelbottom=False, length=0)
#    plt.title(f"{i}\n{tcode}", fontsize=10)
#    sp.axes.grid(False)
#  fp = pathlib.Path(output_filename + ".pdf")
#  plt.savefig(fp, bbox_inches="tight")
#  plt.close()


def argmax_unique(arr, axis):
  arrm = np.argmax(arr, axis)
  arrs = np.sum(arr, axis)
  uni_argmax = (arrs == 1)
  nonunique_mask = np.ma.make_mask((arrs == 1) == False)
  uni_argmax = np.ma.masked_array(arrm, mask=nonunique_mask, fill_value=-1)
  return uni_argmax, nonunique_mask


def make_solver_visualizers(filename, wave, decode_patterns=None, pattern_catalog=None, tile_catalog=None, tile_size=[1, 1]):
    print(wave.shape)
    pattern_total_count = wave.shape[0]
    resolution_order = np.zeros(wave.shape[1:]) # pattern_wave = when was this resolved?
    pattern_solution = np.full(wave.shape[1:], np.nan) # what is the resolved result?
    resolution_method = np.zeros(wave.shape[1:]) # did we set this via observation or propagation?
    choice_count = 0
    vis_count = 0
    max_choices = 140
    def choice_vis(pattern, i, j, wave=None):
        #print(f"choice_vis: {pattern} {i},{j}")
        nonlocal choice_count
        nonlocal resolution_order
        nonlocal resolution_method
        choice_count += 1
        #print(choice_count)
        resolution_order[i][j] = choice_count
        pattern_solution[i][j] = pattern
        resolution_method[i][j] = 2
        #figure_solver_data(f"visualization/{filename}_choice_{choice_count}.png", "order of resolution", resolution_order, 0, max_choices, "gist_ncar")
        #figure_solver_data(f"visualization/{filename}_solution_{choice_count}.png", "chosen pattern", pattern_solution, 0, pattern_total_count, "viridis")
        #figure_solver_data(f"visualization/{filename}_resolution_{choice_count}.png", "resolution method", resolution_method, 0, 2, "inferno")
        if wave:
            assigned_patterns, nonunique_mask = argmax_unique(wave, 0)
            resolved_by_propagation = np.ma.mask_or(nonunique_mask, resolution_method != 0) == 0
            resolution_method[resolved_by_propagation] = 1
            resolution_order[resolved_by_propagation] = choice_count  
            #figure_solver_data(f"visualization/{filename}_wave_{choice_count}.png", "patterns remaining", np.count_nonzero(wave > 0, axis=0), 0, wave.shape[0], "plasma")
              
        
        
        
        
    def wave_vis(wave):
        nonlocal vis_count
        nonlocal resolution_method
        nonlocal resolution_order
        vis_count += 1
        pattern_left_count = np.count_nonzero(wave > 0, axis=0)
        assigned_patterns, nonunique_mask = argmax_unique(wave, 0)
        resolved_by_propagation = np.ma.mask_or(nonunique_mask, resolution_method != 0) == 0
        resolution_method[resolved_by_propagation] = 1
        resolution_order[resolved_by_propagation] = choice_count
        #figure_wave_patterns(filename, pattern_left_count, pattern_total_count)
        #figure_solver_data(f"visualization/{filename}_wave_patterns_{choice_count}.png", "patterns remaining", pattern_left_count, 0, pattern_total_count, "magma")
        if decode_patterns and pattern_catalog and tile_catalog:
            solution_as_ids = np.vectorize(lambda x : decode_patterns[x])(np.argmax(wave,0))
            solution_tile_grid = pattern_grid_to_tiles(solution_as_ids, pattern_catalog)
            #figure_solver_data(f"visualization/{filename}_tiles_assigned_{choice_count}.png", "tiles assigned", solution_tile_grid, 0, pattern_total_count, "plasma")
            img = tile_grid_to_image(solution_tile_grid, tile_catalog, tile_size)
            #figure_solver_image(f"visualization/{filename}_solution_partial_{choice_count}.png", "solved_tiles", img.astype(np.uint8))
            #imageio.imwrite(f"visualization/{filename}_solution_partial_img_{choice_count}.png", img.astype(np.uint8))
            fig_list = [
              {"title": "order of resolution", "data": resolution_order, "vmin": 0, "vmax": max_choices, "cmap": "gist_ncar", "datatype":"figure"},
              {"title": "chosen pattern", "data": pattern_solution, "vmin": 0, "vmax": pattern_total_count, "cmap": "viridis", "datatype":"figure"},
              {"title": "resolution method", "data": resolution_method, "vmin": 0, "vmax": 2, "cmap": "inferno", "datatype":"figure"},   
              {"title": "patterns remaining", "data": pattern_left_count, "vmin": 0, "vmax": pattern_total_count, "cmap": "magma", "datatype":"figure"},
              {"title": "tiles assigned", "data": solution_tile_grid, "vmin": 0, "vmax": pattern_total_count, "cmap": "plasma", "datatype":"figure"},
              {"title": "solved tiles", "data": img.astype(np.uint8), "datatype":"image"}
             ]
            figure_unified("Solver Readout", f"visualization/{filename}_readout_{choice_count:03}.png", fig_list)
      
    return choice_vis, wave_vis

def figure_unified(figure_name_overall, filename, data):
    matfig, axs = plt.subplots(1, len(data), sharey='row', gridspec_kw={'hspace':0, 'wspace':0})
    #matfig = plt.figure(figsize=(16,16))
    #plt.title(f"{figure_name_overall}", fontsize=14, fontweight='bold', y = 0.6)

    for idx, data_obj in enumerate(data):
      if "image" == data[idx]["datatype"]:
        axs[idx].imshow(data[idx]["data"], interpolation='nearest')
      else:
        axs[idx].matshow(data[idx]["data"], vmin=data[idx]["vmin"], vmax=data[idx]["vmax"], cmap=data[idx]["cmap"])
      axs[idx].get_xaxis().set_visible(False)
      axs[idx].get_yaxis().set_visible(False)
      axs[idx].label_outer()

                
    

    #plt.title(data[0]["title"])
    #plt.grid(None)
    #plt.grid(None)
    #ax = plt.gca()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)

    #plt.subplot(1, 2, 1)
    #plt.title(data[1]["title"])
    #plt.matshow(data[1]["data"], vmin=data[1]["vmin"], vmax=data[1]["vmax"], cmap=data[1]["cmap"])
    #plt.grid(None)
    #plt.grid(None)
    #ax = plt.gca()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    

    # for idx, data_obj in enumerate(data): 
    #     plt.subplot(1, len(data), idx+1)
    #     plt.title(data_obj["title"])
    #     plt.matshow(data_obj["data"], vmin=data_obj["vmin"], vmax=data_obj["vmax"], cmap=data_obj["cmap"])
    #     plt.grid(None)
    #     plt.grid(None)
    #     ax = plt.gca()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

      
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close(fig=matfig)
    plt.close('all')
    

vis_count = 0
def visualize_solver(wave):
  pattern_left_count = np.count_nonzero(wave > 0, axis=0)
  pattern_total_count = wave.shape[0]
  figure_wave_patterns(pattern_left_count, pattern_total_count)
  

def make_figure_solver_image(plot_title, img):
    visfig = plt.figure(figsize=(4,4), edgecolor='k', frameon=True)
    plt.imshow(img, interpolation='nearest')
    plt.title(plot_title)
    plt.grid(None)
    plt.grid(None)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 
    return visfig
  
  
def figure_solver_image(filename, plot_title, img):
  #visfig = plt.figure(figsize=(4,4), edgecolor='k', frameon=True)
  #plt.imshow(img, interpolation='nearest')
  #plt.title(plot_title)
  #plt.grid(None)
  #plt.grid(None)
  #ax = plt.gca()
  #ax.get_xaxis().set_visible(False) 
  #ax.get_yaxis().set_visible(False)
  visfig = make_figure_solver_image(plot_title, img)
  plt.savefig(filename, bbox_inches="tight", pad_inches=0)
  plt.close(fig=visfig)
  plt.close('all')

def make_figure_solver_data_fn(plot_title, data, min_count, max_count, cmap_name):
  def data_fn(sub):
      if sub is None:
          print(sub)
          return
      plt.title(plot_title)
      plt.matshow(data, vmin=min_count, vmax=max_count, cmap=cmap_name)
      plt.grid(None)
      plt.grid(None)
      ax = plt.gca()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  
  return data_fn
  
def make_figure_solver_data(plot_title, data, min_count, max_count, cmap_name):
  visfig = plt.figure(figsize=(4,4), edgecolor='k', frameon=True)
  plt.title(plot_title)
  plt.matshow(data, vmin=min_count, vmax=max_count, cmap=cmap_name)
  plt.grid(None)
  plt.grid(None)
  ax = plt.gca()
  ax.get_xaxis().set_visible(False) 
  ax.get_yaxis().set_visible(False)
  return visfig
  
  
def figure_solver_data(filename, plot_title, data, min_count, max_count, cmap_name):
    # visfig = plt.figure(figsize=(4,4), edgecolor='k', frameon=True)
    # plt.title(plot_title)
    # plt.matshow(data, vmin=min_count, vmax=max_count, cmap=cmap_name)
    # plt.grid(None)
    # plt.grid(None)
    # ax = plt.gca()
    # ax.get_xaxis().set_visible(False) 
    # ax.get_yaxis().set_visible(False)
    visfig = make_figure_solver_data(plot_title, data, min_count, max_count, cmap_name)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close(fig=visfig)
    plt.close('all')




  
def figure_wave_patterns(filename, pattern_left_count, max_count):
  global vis_count
  vis_count += 1
  visfig = plt.figure(figsize=(4,4), edgecolor='k', frameon=True)
  #plt.title(f"solver")
  
  #ax = plt.subplot(1,5,1)
  plt.title("wave")
  plt.matshow(pattern_left_count, vmin=0, vmax=max_count, cmap="plasma")
  plt.grid(None)
  #plt.set_yticklabels([])
  #plt.set_xticklabels([])
  plt.grid(None)
  plt.savefig(f"{filename}_wave_patterns_{vis_count}.png")
  plt.close(fig=visfig)


 
  #assert False
  #1. improving student success
  # graduate states, first year retention, achievement gaps (rates of graduation for first 1st gen, low income, underrepresented), challenges to support graduate students
  # we have to to work together to figure out solutions
  #2. make a climate that is inclusive, diverse, and welcoming
  # a signifigant number of students feel like they don't belong, correlated with 1st/low/under groups
  # inclusivity is thinking about the climate of the campus
  # talked about when EVC at riverside
  # microclimate is the people you expereince throughout your day
  # giving people the benefit of the doubt
  # affect our processes and procedures that get in your way (e.g. paying fees by credit card)
  #3. to continue to bolster our research profile
  # AAU
  # 
  
def tile_grid_to_image(tile_grid, tile_catalog, tile_size, visualize=False, partial=False, color_channels=3):
    """
    Takes a tile_grid and transforms it into an image, using the information
    in tile_catalog. We use tile_size to figure out the size the new image
    should be, and visualize for displaying partial tile patterns.
    """
    new_img = np.zeros((tile_grid.shape[0] * tile_size[0], tile_grid.shape[1] * tile_size[1], color_channels), dtype=np.int64)
    if partial and (len(tile_grid.shape)) > 2:
        for i in range(tile_grid.shape[0]):
            for j in range(tile_grid.shape[1]):
              tiles = tile_grid[i,j]
              print(tiles)      
        pass # TODO: implement rendering partially completed solution
        assert False
    else:
        for i in range(tile_grid.shape[0]):
            for j in range(tile_grid.shape[1]):
                tile = tile_grid[i,j]
                for u in range(tile_size[0]):
                    for v in range(tile_size[1]):
                        pixel = [200, 0, 200]
                        ## If we want to display a partial pattern, it is helpful to
                        ## be able to show empty cells. Therefore, in visualize mode,
                        ## we use -1 as a magic number for a non-existant tile.
                        if visualize and ((-1 == tile) or (-2 == tile)):
                            if (-1 == tile):
                                if 0 == (i + j) % 2:
                                    pixel = [255, 0, 255]
                            if (-2 == tile):
                                pixel = [0, 255, 255]
                        else:
                            pixel = tile_catalog[tile][u,v]
                        # TODO: will need to change if using an image with more than 3 channels
                        new_img[(i * tile_size[0]) + u, (j * tile_size[1]) + v] = np.resize(pixel, new_img[(i * tile_size[0]) + u, (j * tile_size[1]) + v].shape)
    return new_img


def figure_list_of_tiles(unique_tiles, tile_catalog, output_filename="list_of_tiles"):
  plt.figure(figsize=(4,4), edgecolor='k', frameon=True)
  plt.title('Extracted Tiles')
  s = math.ceil(math.sqrt(len(unique_tiles)))+1
  for i,tcode in enumerate(unique_tiles[0]):
    sp = plt.subplot(s, s, i + 1).imshow(tile_catalog[tcode])
    sp.axes.tick_params(labelleft=False, labelbottom=False, length=0)
    plt.title(f"{i}\n{tcode}", fontsize=10)
    sp.axes.grid(False)
  fp = pathlib.Path(output_filename + ".pdf")
  plt.savefig(fp, bbox_inches="tight")
  plt.close()

def figure_false_color_tile_grid(tile_grid, output_filename="./false_color_tiles"):
    figure_plot = plt.matshow(tile_grid, cmap='gist_ncar',extent=(0, tile_grid.shape[1], tile_grid.shape[0], 0))
    plt.title('False Color Map of Tiles in Input Image');
    figure_plot.axes.grid(None)
    plt.savefig(output_filename + ".png", bbox_inches="tight")
    plt.close()
    
def figure_tile_grid(tile_grid, tile_catalog, tile_size):
  img = tile_grid_to_image(tile_grid, tile_catalog, tile_size)
  


def render_pattern(render_pattern, tile_catalog):
  """Turn a pattern into an image"""
  rp_iter = np.nditer(render_pattern, flags=['multi_index'])
  output = np.zeros(render_pattern.shape + (3,), dtype=np.uint32)
  while not rp_iter.finished:
    # Note that this truncates images with more than 3 channels down to just the channels in the output.
    # If we want to have alpha channels, we'll need a different way to handle this.
    output[rp_iter.multi_index] = np.resize(tile_catalog[render_pattern[rp_iter.multi_index]], output[rp_iter.multi_index].shape)
    rp_iter.iternext()
  return output

def figure_pattern_catalog(pattern_catalog, tile_catalog, pattern_weights, pattern_width, output_filename="pattern_catalog"):
    s_columns = 24 // min(24, pattern_width)
    s_rows = 1 + (int(len(pattern_catalog)) // s_columns)
    fig = plt.figure(figsize=(s_columns, s_rows * 1.5))
    plt.title('Extracted Patterns')
    counter = 0
    for i, tcode in pattern_catalog.items():
        pat_cat = pattern_catalog[i]
        ptr = render_pattern(pat_cat, tile_catalog).astype(np.uint8)
        sp = plt.subplot(s_rows, s_columns, counter + 1)
        spi = sp.imshow(ptr)
        spi.axes.xaxis.set_label_text(f'({pattern_weights[i]})')
        sp.set_title(f"{counter}\n{i}", fontsize=3)
        spi.axes.tick_params(labelleft=False,labelbottom=False, left=False, bottom=False)
        spi.axes.grid(False)
        counter += 1
    plt.savefig(output_filename + "_patterns.pdf", bbox_inches="tight")
    plt.close()


def render_tiles_to_output(tile_grid, tile_catalog, tile_size, output_filename):
  img = tile_grid_to_image(tile_grid, tile_catalog, tile_size)
  imageio.imwrite(output_filename, img.astype(np.uint8))


def blit(destination, sprite, upper_left, layer = False, check=False):
    """
    Blits one multidimensional array into another numpy array.
    """
    lower_right = [((a + b) if ((a + b) < c) else c) for a,b,c in zip(upper_left, sprite.shape, destination.shape)]
    if min(lower_right) < 0:
        return
    
    for i_index, i in enumerate(range(upper_left[0], lower_right[0])):
        for j_index, j in enumerate(range(upper_left[1], lower_right[1])):
            if (i >= 0) and (j >= 0):
                if len(destination.shape) > 2:
                    destination[i, j, layer] = sprite[i_index, j_index]
                else:
                    if check:
                        if (destination[i, j] == sprite[i_index, j_index]) or (destination[i, j] == -1) or {sprite[i_index, j_index] == -1}:
                            destination[i, j] = sprite[i_index, j_index]
                        else:
                            print("ERROR, mismatch: destination[{i},{j}] = {destination[i, j]}, sprite[{i_index}, {j_index}] = {sprite[i_index, j_index]}")
                    else:
                        destination[i, j] = sprite[i_index, j_index]
    return destination

class InvalidAdjacency(Exception):
  """The combination of patterns and offsets results in pattern combinations that don't match."""
  pass
  
def validate_adjacency(pattern_a, pattern_b, preview_size, upper_left_of_center, adj_rel):
  preview_adj_a_first = np.full((preview_size, preview_size), -1, dtype=np.int64)
  preview_adj_b_first = np.full((preview_size, preview_size), -1, dtype=np.int64)      
  blit(preview_adj_b_first, pattern_b,
       (upper_left_of_center[1] + adj_rel[0][1],
        upper_left_of_center[0] + adj_rel[0][0]), check=True)
  blit(preview_adj_b_first, pattern_a, upper_left_of_center, check=True)
  
  blit(preview_adj_a_first, pattern_a, upper_left_of_center, check=True)
  blit(preview_adj_a_first, pattern_b,
       (upper_left_of_center[1] + adj_rel[0][1],
        upper_left_of_center[0] + adj_rel[0][0]), check=True)
  if not np.array_equiv(preview_adj_a_first, preview_adj_b_first):
    print(adj_rel)
    print(pattern_a)
    print(pattern_b)
    print(preview_adj_a_first)
    print(preview_adj_b_first)
    raise InvalidAdjacency

  
  
def figure_adjacencies(adjacency_relations_list, adjacency_directions, tile_catalog, patterns, pattern_width, tile_size, output_filename="adjacency", render_b_first=False):
#    try:
        adjacency_directions_list = list(dict(adjacency_directions).values())
        figadj = plt.figure(figsize=(12,1+len(adjacency_relations_list[:64])), edgecolor='b')
        plt.title('Adjacencies')
        max_offset = max([abs(x) for x in list(itertools.chain.from_iterable(adjacency_directions_list))])

        for i,adj_rel in enumerate(adjacency_relations_list[:64]):
            preview_size = (pattern_width + max_offset * 2)
            preview_adj = np.full((preview_size, preview_size), -1, dtype=np.int64)    
            upper_left_of_center = [max_offset,max_offset]

            pattern_a = patterns[adj_rel[1]]
            pattern_b = patterns[adj_rel[2]]
            validate_adjacency(pattern_a, pattern_b, preview_size, upper_left_of_center, adj_rel)
            if render_b_first:
              blit(preview_adj, pattern_b,
                 (upper_left_of_center[1] + adj_rel[0][1], 
                  upper_left_of_center[0] + adj_rel[0][0]), check=True)
              blit(preview_adj, pattern_a, upper_left_of_center, check=True)
            else:
              blit(preview_adj, pattern_a, upper_left_of_center, check=True)
              blit(preview_adj, pattern_b,
                   (upper_left_of_center[1] + adj_rel[0][1], 
                    upper_left_of_center[0] + adj_rel[0][0]), check=True)

            ptr = tile_grid_to_image(preview_adj, tile_catalog, tile_size, visualize=True).astype(np.uint8)
            
            subp = plt.subplot(math.ceil(len(adjacency_relations_list[:64]) / 4),4, i+1)
            spi = subp.imshow(ptr)
            spi.axes.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            plt.title(f'{i}:\n({adj_rel[1]} +\n{adj_rel[2]})\n by {adj_rel[0]}', fontsize=10)
            
            indicator_rect = matplotlib.patches.Rectangle((upper_left_of_center[1] - 0.51, upper_left_of_center[0] - 0.51), pattern_width, pattern_width, Fill=False, edgecolor='b', linewidth=3.0, linestyle=':')
            
            spi.axes.add_artist(indicator_rect)
            spi.axes.grid(False)
        plt.savefig(output_filename + "_adjacency.pdf", bbox_inches="tight")
        plt.close()
#    except ValueError as e:
#        print(e)

  

# def render_patterns_to_output(pattern_grid, pattern_catalog, tile_catalog, tile_size, output_filename):
#   tile_grid = pattern_grid_to_tiles(pattern_grid, pattern_catalog)
#   img = tile_grid_to_image(tile_grid, tile_catalog, tile_size)
#   imageio.imwrite(output_filename, img)

    
# def render_patterns_to_output(output_tile_grid, wave_table, partial=False, visualize=True):
#     pattern_grid = np.array(output_tile_grid, dtype=np.int64)
#     has_gaps = np.any(np.count_nonzero(wave_table, axis=2) != 1) 
#     if has_gaps:
#         pattern_grid = np.array(partial_output_grid, dtype=np.int64)
#     render_grid = np.full(pattern_grid.shape,  WFC_PARTIAL_BLANK, dtype=np.int64)
#     pattern_center = wfc_state.wfc_ns.pattern_center
#     for row in range(wfc_state.rows):
#         if WFC_DEBUGGING:
#             print()
#         for column in range(wfc_state.columns):
#             if (len(pattern_grid.shape) > 2):
#                 if WFC_DEBUGGING:
#                     print('[',end='')
#                 pattern_list = []
#                 for z in range(wfc_state.number_of_patterns):
#                     pattern_list.append(pattern_grid[(row,column,z)])
#                 pattern_list = [pattern_grid[(row,column,z)] for z in range(wfc_state.number_of_patterns) if (pattern_grid[(row,column,z)] != -1) and (pattern_grid[(row,column,z)] != WFC_NULL_VALUE)]
#                 for pl_count, the_pattern in enumerate(pattern_list):
#                     if WFC_DEBUGGING:
#                         print(the_pattern, end='')
#                     the_pattern_tiles = wfc_state.wfc_ns.pattern_catalog[the_pattern][pattern_center[0]:pattern_center[0]+1,pattern_center[1]:pattern_center[1]+1]
#                     if WFC_DEBUGGING:
#                         print(the_pattern_tiles, end=' ')
#                     render_grid = blit(render_grid, the_pattern_tiles, (row,column), layer = pl_count)
#                 if WFC_DEBUGGING:
#                     print(']',end=' ')
#             else:
#                 if WFC_DEBUGGING:
#                     print(pattern_grid[(row,column)], end=',')
#                 if WFC_NULL_VALUE != pattern_grid[(row,column)]:
#                     the_pattern = wfc_state.wfc_ns.pattern_catalog[pattern_grid[(row,column)]]
#                     p_x = wfc_state.wfc_ns.pattern_center[0]
#                     p_y = wfc_state.wfc_ns.pattern_center[1]
#                     the_pattern = the_pattern[p_x:p_x+1, p_y:p_y+1]
#                     render_grid = blit(render_grid, the_pattern, (row, column))
#     if WFC_DEBUGGING:
#         print("\nrender grid")
#         print(render_grid)
#     ptr = tiles_to_images(wfc_state.wfc_ns, render_grid, wfc_state.wfc_ns.tile_catalog, wfc_state.wfc_ns.tile_size, visualize=True, partial=partial).astype(np.uint8)
#     if WFC_DEBUGGING:
#         print(f"ptr {ptr}")
        
#     if visualize:
#         fig, ax = subplots(figsize=(16,16))
#         #ax.grid(color="magenta", linewidth=1.5)
#         ax.tick_params(direction='in', bottom=False, left=False)

#         im = ax.imshow(ptr)
#         for axis, dim in zip([ax.xaxis, ax.yaxis],[wfc_state.columns, wfc_state.rows]):
#             axis.set_ticks(np.arange(-0.5, dim + 0.5, 1))
#             axis.set_ticklabels([])
#     #print(ptr)
#     imageio.imwrite(wfc_state.wfc_ns.output_filename, ptr)



#def figure_adjacencies(adjacency_relations_list, pattern_catalog, tile_catalog):
#  print(adjacency_relations_list)
#  return
    
# def figure_adjacencies(wfc_ns, adjacency_relations_list):
#     try:
#         figadj = figure(figsize=(12,1+len(adjacency_relations_list)), edgecolor='b')
#         title('Adjacencies')
#         max_offset = max([abs(x) for x in list(itertools.chain.from_iterable(wfc_ns.adjacency_directions.values()))])

#         for i,adj_rel in enumerate(adjacency_relations_list):
#             preview_size = (wfc_ns.pattern_width + max_offset*2)
#             preview_adj = np.full((preview_size,preview_size), -1, dtype=np.int64)    
#             upper_left_of_center = CoordXY(x=max_offset,y=max_offset)#(ns.pattern_width, ns.pattern_width)
#             #print(f"adj_rel: {adj_rel}")
#             blit(preview_adj, wfc_ns.patterns[adj_rel[1]], upper_left_of_center, check=True)
#             blit(preview_adj, wfc_ns.patterns[adj_rel[2]], 
#                  (upper_left_of_center.y + wfc_ns.adjacency_directions[adj_rel[0]].y, 
#                   upper_left_of_center.x + wfc_ns.adjacency_directions[adj_rel[0]].x), check=True)

#             ptr = tiles_to_images(wfc_ns, preview_adj, wfc_ns.tile_catalog, wfc_ns.tile_size, visualize=True).astype(np.uint8)
            
#             subp = subplot(math.ceil(len(adjacency_relations_list) / 4),4, i+1)
#             spi = subp.imshow(ptr)
#             spi.axes.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
#             title(f'{i}: ({adj_rel[1]} + {adj_rel[2]}) by\n{wfc_ns.adjacency_directions[adj_rel[0]]}', fontsize=10)
            
#             indicator_rect = matplotlib.patches.Rectangle((upper_left_of_center.y - 0.51, upper_left_of_center.x - 0.51), wfc_ns.pattern_width, wfc_ns.pattern_width, Fill=False, edgecolor='b', linewidth=3.0, linestyle=':')
            
#             spi.axes.add_artist(indicator_rect)
#             spi.axes.grid(False)
#         plt.savefig(wfc_ns.output_filename + "_adjacency.pdf", bbox_inches="tight")
#         plt.close()
#     except ValueError as e:
#         print(e)
