"Visualize the patterns into tiles and so on."

import matplotlib.pyplot as plt
import numpy as np
import imageio
import math
import pathlib


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
                            pass
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
    



def render_pattern(render_pattern, tile_catalog):
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
        sp.set_title(f"{counter}")
        spi.axes.tick_params(labelleft=False,labelbottom=False, left=False, bottom=False)
        spi.axes.grid(False)
        counter += 1
    plt.savefig(output_filename + "_patterns.pdf", bbox_inches="tight")
    plt.close()

