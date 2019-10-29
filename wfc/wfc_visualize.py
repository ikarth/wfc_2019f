"Visualize the patterns into tiles and so on."

import matplotlib.pyplot as plt
import imageio



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



def image_to_disk(image, filename):
    return


def figure_list_of_tiles(unique_tiles, tile_catalog, tile_ids):
    figure(figsize=(4,4), edgecolor='k', frameon=True)
    title('Extracted Tiles')
    s = math.ceil(math.sqrt(len(unique_tiles)))+1
    for tcode,i in tile_ids.items():
        sp = subplot(s, s, i + 1).imshow(tile_catalog[tcode])
        sp.axes.tick_params(labelleft=False, labelbottom=False, length=0)
        title(i, fontsize=10)
        sp.axes.grid(False)

    plt.close()

def figure_false_color_tile_grid(tile_grid):
    figure_plot = matshow(tile_grid, cmap='gist_ncar',extent=(0, tile_grid.shape[1], tile_grid.shape[0], 0))
    title('False Color Map of Tiles in Input Image');
    figure_plot.axes.grid(None)
    p1t.close()
    
