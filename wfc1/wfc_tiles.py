import wfc.wfc_utilities
from wfc.wfc_utilities import WFC_PARTIAL_BLANK, WFC_NULL_VALUE
from wfc.wfc_utilities import CoordXY, CoordRC, hash_downto, find_pattern_center
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplot, subplots, title, matshow
import math
import numpy as np
import logging

import logging

logging.basicConfig(level=logging.INFO)
wfc_logger = logging.getLogger()

## Helper functions
RGB_CHANNELS = 3


def rgb_to_int(rgb_in):
    """"Takes RGB triple, returns integer representation."""
    return struct.unpack(
        "I", struct.pack("<" + "B" * 4, *(rgb_in + [0] * (4 - len(rgb_in))))
    )[0]


def int_to_rgb(val):
    return [x for x in val.to_bytes(RGB_CHANNELS, "little")]


# In[8]:


import imageio


def load_source_image(filename):
    return imageio.imread(filename)


def image_to_tiles(img, tile_size):
    """
    Takes an images, divides it into tiles, return an array of tiles.
    >>> image_to_tiles(test_ns.img, test_ns.tile_size)
    array([[[[[255, 255, 255]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[255, 255, 255]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[255, 255, 255]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[255, 255, 255]]]],
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
           [[[[255, 255, 255]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[  0,   0,   0]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[  0,   0,   0]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[  0,   0,   0]]]],
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
           [[[[255, 255, 255]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[  0,   0,   0]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[255,   0,   0]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[  0,   0,   0]]]],
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
           [[[[255, 255, 255]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[  0,   0,   0]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[  0,   0,   0]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[  0,   0,   0]]]]], dtype=uint8)
    """
    padding_argument = [(0, 0), (0, 0), (0, 0)]
    for input_dim in [0, 1]:
        padding_argument[input_dim] = (
            0,
            (tile_size - img.shape[input_dim]) % tile_size,
        )
    img = np.pad(img, padding_argument, mode="constant")
    tiles = img.reshape(
        (
            img.shape[0] // tile_size,
            tile_size,
            img.shape[1] // tile_size,
            tile_size,
            img.shape[2],
        )
    ).swapaxes(1, 2)
    return tiles


def tile_to_image(tile, tile_catalog, tile_size, visualize=False):
    """
    Takes a single tile and returns the pixel image representation.
    """
    new_img = np.zeros((tile_size, tile_size, 3), dtype=np.int64)
    for u in range(tile_size):
        for v in range(tile_size):
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
                    pixel = tile_catalog[tile][u, v]
            new_img[u, v] = pixel


def tiles_to_images(
    wfc_ns,
    tile_grid,
    tile_catalog,
    tile_size,
    visualize=False,
    partial=False,
    grid_count=None,
):
    """
    Takes a tile_grid and transforms it into an image, using the information
    in tile_catalog. We use tile_size to figure out the size the new image
    should be, and visualize for displaying partial tile patterns.
    """
    new_img = np.zeros(
        (
            tile_grid.shape[0] * tile_size,
            tile_grid.shape[1] * tile_size,
            wfc_ns.channels,
        ),
        dtype=np.int64,
    )
    if partial and (len(tile_grid.shape) > 2):
        for i in range(tile_grid.shape[0]):
            for j in range(tile_grid.shape[1]):
                for u in range(wfc_ns.tile_size):
                    for v in range(wfc_ns.tile_size):
                        pixel_merge_list = []
                        for k in range(tile_grid.shape[2]):
                            tile = tile_grid[i, j, k]
                            ## If we want to display a partial pattern, it is helpful to
                            ## be able to show empty cells. Therefore, in visualize mode,
                            ## we use -1 as a magic number for a non-existant tile.
                            pixel = None  # [200, 0, 200]
                            # print(tile)
                            if (visualize) and ((-1 == tile) or (-2 == tile)):
                                if -1 == tile:
                                    pixel = [200, 0, 200]
                                    if 0 == (i + j) % 2:
                                        pixel = [255, 0, 255]
                                else:
                                    pixel = [0, 255, 255]
                            else:
                                if (WFC_PARTIAL_BLANK != tile) and (
                                    WFC_NULL_VALUE != tile
                                ):  # TODO: instead of -3, use MaskedArrays
                                    pixel = tile_catalog[tile][u, v]
                            if not (pixel is None):
                                pixel_merge_list.append(pixel)
                        if len(pixel_merge_list) == 0:
                            if 0 == (i + j) % 2:
                                pixel_merge_list.append([255, 0, 255])
                            else:
                                pixel_merge_list.append([0, 172, 172])

                        if len(pixel_merge_list) > 0:
                            pixel_to_add = pixel_merge_list[0]
                            if len(pixel_merge_list) > 1:
                                pixel_to_add = [
                                    round(sum(x) / len(pixel_merge_list))
                                    for x in zip(*pixel_merge_list)
                                ]
                            try:
                                while len(pixel_to_add) < wfc_ns.channels:
                                    pixel_to_add.append(255)
                                new_img[
                                    (i * wfc_ns.tile_size) + u,
                                    (j * wfc_ns.tile_size) + v,
                                ] = pixel_to_add
                            except TypeError as e:
                                wfc_logger.warning(e)
                                wfc_logger.warning(
                                    "Tried to add {} from {}".format(
                                        pixel_to_add, pixel_merge_list
                                    )
                                )
    else:
        for i in range(tile_grid.shape[0]):
            for j in range(tile_grid.shape[1]):
                tile = tile_grid[i, j]
                for u in range(wfc_ns.tile_size):
                    for v in range(wfc_ns.tile_size):
                        ## If we want to display a partial pattern, it is helpful to
                        ## be able to show empty cells. Therefore, in visualize mode,
                        ## we use -1 as a magic number for a non-existant tile.
                        pixel = [200, 0, 200]
                        # print(f"tile: {tile}")
                        if (visualize) and ((-1 == tile) or (-2 == tile)):
                            if -1 == tile:
                                if 0 == (i + j) % 2:
                                    pixel = [255, 0, 255]
                            if -2 == tile:
                                pixel = [0, 255, 255]
                        else:
                            if WFC_PARTIAL_BLANK != tile:
                                pixel = tile_catalog[tile][u, v]
                        # Watch out for images with more than 3 channels!
                        new_img[
                            (i * wfc_ns.tile_size) + u, (j * wfc_ns.tile_size) + v
                        ] = np.resize(
                            pixel,
                            new_img[
                                (i * wfc_ns.tile_size) + u, (j * wfc_ns.tile_size) + v
                            ].shape,
                        )
    logging.debug("Output image shape is", new_img.shape)
    return new_img


# Past this point, WFC itself doesn't care about the exact content of the image, just that it exists. So we're going to pack all that information away behind some data structures: a dictionary of the tiles and a matrix with the tiles in the input image. The `tile_grid` is the most important thing for automatically figuring out adjacencies, while the `tile_catalog` is what will use at the end to render the final results.
#
# Here's the default tile cataloger. Take the big bag of tiles that we've got and arrange them in a dictionary that categorizes similar tiles under the same key.
#
# `tile_catalog`:  dictionary to translate the hash-key ID of the tile to the image representation of the tile. We won't need this again until we do the output render.
# `tile_grid`:  the original input image, only this time expressed as a 2D array of tile IDs.
# `code_list`: 1D array of the tile IDs.
# `unique_tiles`:  1D array of the unique tiles in the tile grid.
#
# You can modify this to make your own tile cataloger, with different behavor. For example, if you wanted each tile to have its own id even if it had the same image, or contrariwise if you wanted to group all of the background tiles under the same heading.
#

# In[9]:


def make_tile_catalog(nspace):
    """
    """
    tiles = image_to_tiles(nspace.img, nspace.tile_size)
    logging.info(f"The shape of the input image is {tiles.shape}")
    # print(f'The shape of the input image is {tiles.shape}')
    # print(tiles)
    print(
        (
            tiles.shape[0] * tiles.shape[1],
            nspace.tile_size,
            nspace.tile_size,
            nspace.channels,
        )
    )

    tile_list = np.array(tiles).reshape(
        (
            tiles.shape[0] * tiles.shape[1],
            nspace.tile_size,
            nspace.tile_size,
            nspace.channels,
        )
    )
    ## Make Tile Catalog
    code_list = np.array(hash_downto(tiles, 2)).reshape(
        (tiles.shape[0] * tiles.shape[1])
    )
    tile_grid = np.array(hash_downto(tiles, 2), dtype=np.int64)
    unique_tiles = np.unique(tile_grid, return_counts=True)

    tile_catalog = {}
    for i, j in enumerate(code_list):
        tile_catalog[j] = tile_list[i]
    return tile_catalog, tile_grid, code_list, unique_tiles


# Let's visualize what we have so far. We'll load an image, turn it into tiles, and then render those tiles into an image. If everything is working right, we should have two identical images.

# In[10]:


def show_input_to_output(img_ns):
    """
    Does the input equal the output?    

    >>> [show_input_to_output(test_ns), load_source_image(test_ns.img_filename)]
    [[[255 255 255]
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    <BLANKLINE>
     [[255 255 255]
      [  0   0   0]
      [  0   0   0]
      [  0   0   0]]
    <BLANKLINE>
     [[255 255 255]
      [  0   0   0]
      [255   0   0]
      [  0   0   0]]
    <BLANKLINE>
     [[255 255 255]
      [  0   0   0]
      [  0   0   0]
      [  0   0   0]]]
    [None, Image([[[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    <BLANKLINE>
           [[255, 255, 255],
            [  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0]],
    <BLANKLINE>
           [[255, 255, 255],
            [  0,   0,   0],
            [255,   0,   0],
            [  0,   0,   0]],
    <BLANKLINE>
           [[255, 255, 255],
            [  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0]]], dtype=uint8)]
    """
    figure()

    sp = subplot(1, 2, 1).imshow(img_ns.img)
    sp.axes.grid(False)
    sp.axes.tick_params(
        bottom=False,
        left=False,
        which="both",
        labelleft=False,
        labelbottom=False,
        length=0,
    )
    title("Input Image", fontsize=10)
    outimg = tiles_to_images(
        img_ns, img_ns.tile_grid, img_ns.tile_catalog, img_ns.tile_size
    )
    sp = subplot(1, 2, 2).imshow(outimg.astype(np.uint8))
    sp.axes.tick_params(
        bottom=False,
        left=False,
        which="both",
        labelleft=False,
        labelbottom=False,
        length=0,
    )
    title("Output Image From Tiles", fontsize=10)
    sp.axes.grid(False)
    # print(outimg.astype(np.uint8))
    # print(img_ns)
    plt.savefig(img_ns.output_filename + "_input_to_output.pdf", bbox_inches="tight")
    plt.close()


def show_extracted_tiles(img_ns):
    figure(figsize=(4, 4), edgecolor="k", frameon=True)
    title("Extracted Tiles")
    s = math.ceil(math.sqrt(len(img_ns.unique_tiles))) + 1
    # print(s)
    for tcode, i in img_ns.tile_ids.items():
        sp = subplot(s, s, i + 1).imshow(img_ns.tile_catalog[tcode])
        sp.axes.tick_params(labelleft=False, labelbottom=False, length=0)
        title(i, fontsize=10)
        sp.axes.grid(False)

    plt.close()


# A nice little diagram of our palette of tiles.
#
# And, just to check that our internal `tile_grid` representation has reasonable values, let's look at it directly. This will be useful if we have to debug the inner workings of our propagator.

# In[11]:


def show_false_color_tile_grid(img_ns):
    pl = matshow(
        img_ns.tile_grid,
        cmap="gist_ncar",
        extent=(0, img_ns.tile_grid.shape[1], img_ns.tile_grid.shape[0], 0),
    )
    title("False Color Map of Tiles in Input Image")
    pl.axes.grid(None)
    # edgecolor('black')


if __name__ == "__main__":
    import types

    test_ns = types.SimpleNamespace(
        img_filename="red_maze.png",
        seed=87386,
        tile_size=1,
        pattern_width=2,
        channels=3,
        adjacency_directions=dict(
            enumerate(
                [
                    CoordXY(x=0, y=-1),
                    CoordXY(x=1, y=0),
                    CoordXY(x=0, y=1),
                    CoordXY(x=-1, y=0),
                ]
            )
        ),
        periodic_input=True,
        periodic_output=True,
        generated_size=(3, 3),
        screenshots=1,
        iteration_limit=0,
        allowed_attempts=1,
    )
    test_ns = wfc_utilities.find_pattern_center(test_ns)
    test_ns = wfc_utilities.load_visualizer(test_ns)
    test_ns.img = load_source_image(test_ns.img_filename)
    (
        test_ns.tile_catalog,
        test_ns.tile_grid,
        test_ns.code_list,
        test_ns.unique_tiles,
    ) = make_tile_catalog(test_ns)
    test_ns.tile_ids = {
        v: k for k, v in dict(enumerate(test_ns.unique_tiles[0])).items()
    }
    test_ns.tile_weights = {
        a: b for a, b in zip(test_ns.unique_tiles[0], test_ns.unique_tiles[1])
    }
    import doctest

    doctest.testmod()
