"Visualize the patterns into tiles and so on."

from wfc_patterns import pattern_grid_to_tiles
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

def tile_grid_to_image(tile_grid, tile_catalog, tile_size, visualize=False, partial=False, color_channels=3):
    """
    Takes a tile_grid and transforms it into an image, using the information
    in tile_catalog. We use tile_size to figure out the size the new image
    should be, and visualize for displaying partial tile patterns.
    """
    new_img = np.zeros((tile_grid.shape[0] * tile_size[0], tile_grid.shape[1] * tile_size[1], color_channels), dtype=np.int64)
    if partial and (len(tile_grid.shape)) > 2:
        pass # TODO: implement rendering partially completed solution
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


def figure_list_of_tiles(unique_tiles, tile_catalog, fname="list_of_tiles.pdf"):
  plt.figure(figsize=(4,4), edgecolor='k', frameon=True)
  plt.title('Extracted Tiles')
  s = math.ceil(math.sqrt(len(unique_tiles)))+1
  for i,tcode in enumerate(unique_tiles[0]):
    sp = plt.subplot(s, s, i + 1).imshow(tile_catalog[tcode])
    sp.axes.tick_params(labelleft=False, labelbottom=False, length=0)
    plt.title(f"{i} : {tcode}", fontsize=10)
    sp.axes.grid(False)
  fp = pathlib.Path(fname)
  plt.savefig(fp, bbox_inches="tight")
  plt.close()

def figure_false_color_tile_grid(tile_grid, filename="./false_color_tiles.png"):
    figure_plot = plt.matshow(tile_grid, cmap='gist_ncar',extent=(0, tile_grid.shape[1], tile_grid.shape[0], 0))
    plt.title('False Color Map of Tiles in Input Image');
    figure_plot.axes.grid(None)
    plt.savefig(filename, bbox_inches="tight")
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

  
def figure_adjacencies(adjacency_relations_list, adjacency_directions, tile_catalog, patterns, pattern_width, tile_size, output_filename="adjacency"):
#    try:
        adjacency_directions_list = dict(adjacency_directions).values()
        figadj = plt.figure(figsize=(12,1+len(adjacency_relations_list)), edgecolor='b')
        plt.title('Adjacencies')
        max_offset = max([abs(x) for x in list(itertools.chain.from_iterable(adjacency_directions_list))])

        for i,adj_rel in enumerate(adjacency_relations_list):
            preview_size = (pattern_width + max_offset * 2)
            preview_adj = np.full((preview_size, preview_size), -1, dtype=np.int64)    
            upper_left_of_center = [max_offset,max_offset]

            blit(preview_adj, patterns[adj_rel[1]], upper_left_of_center, check=True)
            blit(preview_adj, patterns[adj_rel[2]],
                 (upper_left_of_center[1] + adj_rel[0][1], 
                  upper_left_of_center[0] + adj_rel[0][0]), check=True)

            ptr = tile_grid_to_image(preview_adj, tile_catalog, tile_size, visualize=True).astype(np.uint8)
            
            subp = plt.subplot(math.ceil(len(adjacency_relations_list) / 4),4, i+1)
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
