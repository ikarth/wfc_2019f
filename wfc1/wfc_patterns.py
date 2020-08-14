import logging

# from matplotlib.pyplot import figure, subplot, subplots, title, matshow
import wfc.wfc_utilities
from wfc.wfc_utilities import CoordXY, CoordRC, hash_downto, find_pattern_center
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplot, subplots, title, matshow

import logging

logging.basicConfig(level=logging.INFO)
wfc_logger = logging.getLogger()

# # Patterns
#
# We could figure out the valid adjacencies between the tiles by hand and WFC would work just fine---in fact, Gumin's original SimpleTile model is exactly that. But lets bring automation on our side and calculate the adjacencies instead.
#
# To do this, we're going to use Gumin's abstraction of patterns. A pattern is simply a region of tiles, in this case in the original image. Instead of talking about adjacencies between patterns, we will use adjacencies between _patterns_. This lets us automatically capture more expressive information about the source image.
#
# The size of the patterns influences the results. 1x1 patterns behave exactly as tiles. 2x2 and 3x3 are common in generating pixel art. Larger patterns are better at capturing relationships but need more valid samples to retain the same flexibility. Which also leads to larger patterns rapidly getting more and more computationally expensive.

# ## Extracting Patterns

# In[12]:


# def unique_patterns_2d(a, k):
#    assert(k >= 1)
#
#    a = np.pad(a, ((0,k-1),(0,k-1),*(((0,0),)*(len(a.shape)-2))), mode='wrap')
#
#    patches = np.lib.stride_tricks.as_strided(
#        a,
#        (a.shape[0]-k+1,a.shape[1]-k+1,k,k,*a.shape[2:]),
#        a.strides[:2] + a.strides[:2] + a.strides[2:],
#        writeable=False)
#
#    patch_codes = hash_downto(patches,2)


#    uc, ui = np.unique(patch_codes, return_index=True)
#    locs = np.unravel_index(ui, patch_codes.shape)
#    up = patches[locs[0],locs[1]]
#    ids = np.vectorize({c: i for i,c in enumerate(uc)}.get)(patch_codes)

#    return ids, up

# def make_pattern_catalog(img_ns):

#    ns.number_of_rotations = 4# TODO: take this as an argument
#    more_grids = [img_ns.tile_grid]
#    for rotation in range(1, ns.number_of_rotations):
#        rot_a = np.rot90(img_ns.tile_grid, rotation)
#        more_grids.append(rot_a)
#        ref_v = np.flip(rot_a, 0)
#        more_grids.append(ref_v)
#        ref_h = np.flip(rot_a, 1)
#        more_grids.append(ref_h)

#    all_pattern_catalog = {}
#    all_pattern_weights = {}
#    all_patterns = np.zeros((0,img_ns.pattern_width,img_ns.pattern_width))
#    for m_grid in more_grids:
#        print(m_grid.shape)

#        pattern_codes, patterns = unique_patterns_2d(m_grid, img_ns.pattern_width)
#        print(type(pattern_codes), pattern_codes.shape)
#        print(type(patterns), patterns.shape)

#        logging.info(f'{len(patterns)} unique patterns')
#        assert np.array_equal(m_grid, patterns[pattern_codes][:,:,0,0])
#        pattern_catalog = {i:x for i,x in enumerate(patterns)}
#        logging.debug(f'\npattern_codes: {pattern_codes}')
#        pattern_weights = {i:(pattern_codes == i).sum() for i,x in enumerate(patterns)}
#        logging.debug(f'pattern_weights: {pattern_weights}')
#        all_pattern_catalog = {**all_pattern_catalog, **pattern_catalog}
#        all_patterns = np.concatenate(all_patterns, patterns)

#    return all_pattern_catalog, pattern_weights, patterns


# ns.pattern_catalog, ns.pattern_weights, ns.patterns = make_pattern_catalog(ns)


def unique_patterns_2d(a, k, periodic_input):
    assert k >= 1
    if periodic_input:
        a = np.pad(
            a, ((0, k - 1), (0, k - 1), *(((0, 0),) * (len(a.shape) - 2))), mode="wrap"
        )
    else:
        # TODO: implement non-wrapped image handling
        # a = np.pad(a, ((0,k-1),(0,k-1),*(((0,0),)*(len(a.shape)-2))), mode='constant', constant_values=None)
        a = np.pad(
            a, ((0, k - 1), (0, k - 1), *(((0, 0),) * (len(a.shape) - 2))), mode="wrap"
        )

    patches = np.lib.stride_tricks.as_strided(
        a,
        (a.shape[0] - k + 1, a.shape[1] - k + 1, k, k, *a.shape[2:]),
        a.strides[:2] + a.strides[:2] + a.strides[2:],
        writeable=False,
    )
    patch_codes = hash_downto(patches, 2)
    uc, ui = np.unique(patch_codes, return_index=True)
    locs = np.unravel_index(ui, patch_codes.shape)
    up = patches[locs[0], locs[1]]
    ids = np.vectorize({c: i for i, c in enumerate(uc)}.get)(patch_codes)
    wfc_logger.debug(ids)
    return ids, up


def unique_patterns_brute_force(grid, size, periodic_input):
    padded_grid = np.pad(
        grid,
        ((0, size - 1), (0, size - 1), *(((0, 0),) * (len(grid.shape) - 2))),
        mode="wrap",
    )
    patches = []
    for x in range(grid.shape[0]):
        row_patches = []
        for y in range(grid.shape[1]):
            row_patches.append(
                np.ndarray.tolist(padded_grid[x : x + size, y : y + size])
            )
        patches.append(row_patches)
    patches = np.array(patches)
    patch_codes = hash_downto(patches, 2)
    uc, ui = np.unique(patch_codes, return_index=True)
    locs = np.unravel_index(ui, patch_codes.shape)
    up = patches[locs[0], locs[1]]
    ids = np.vectorize({c: i for i, c in enumerate(uc)}.get)(patch_codes)
    wfc_logger.debug(ids)
    return ids, up


def make_pattern_catalog_from_grid(tile_grid, pattern_width, periodic_input):
    pattern_codes, patterns = unique_patterns_2d(
        tile_grid, pattern_width, periodic_input
    )

    VERIFY = False
    if VERIFY:
        pattern_codes2, patterns2 = unique_patterns_brute_force(
            tile_grid, pattern_width, periodic_input
        )
        assert np.array_equal(pattern_codes, pattern_codes2)
        assert np.array_equal(patterns, patterns2)
    logging.info(f"{len(patterns)} unique patterns")

    assert np.array_equal(tile_grid, patterns[pattern_codes][:, :, 0, 0])
    pattern_catalog = {i: x for i, x in enumerate(patterns)}
    logging.debug(f"\npattern_codes: {pattern_codes}")
    pattern_weights = {i: (pattern_codes == i).sum() for i, x in enumerate(patterns)}
    logging.debug(f"pattern_weights: {pattern_weights}")

    return pattern_catalog, pattern_weights, patterns, pattern_codes


def reflect_pattern(pattern):
    # print(f"reflect:\n{pattern}\nto\n{np.fliplr(pattern).copy()}\n")
    return np.fliplr(pattern).copy()


def rotate_pattern(pattern):
    # print(f"rotate:\n{pattern}\nto\n{np.rot90(pattern).copy()}\n")
    return np.rot90(pattern).copy()


def make_pattern_catalog_with_symmetry(
    tile_grid, pattern_width, symmetry, periodic_input
):
    (
        pattern_catalog,
        pattern_weights,
        patterns,
        pattern_grid,
    ) = make_pattern_catalog_from_grid(tile_grid, pattern_width, periodic_input)
    # print(patterns.shape)
    # print('~~~~~~~~~')
    if symmetry > 1:
        for i in list(pattern_catalog.keys()):
            base_pattern = pattern_catalog[i].copy()
            for sym_op in range(2, symmetry + 1):
                # print(f"{sym_op} and {(sym_op % 2 == 0)}")
                r_func = reflect_pattern if (sym_op % 2 == 0) else rotate_pattern
                # print(r_func.__name__)
                new_pattern = r_func(base_pattern)
                try:
                    pattern_index = [
                        np.array_equal(new_pattern, x) for x in pattern_catalog.values()
                    ].index(True)
                    pattern_weights[pattern_index] += 1
                except ValueError:
                    new_index = len(pattern_weights)
                    pattern_catalog[new_index] = new_pattern.copy()
                    pattern_weights[new_index] = 1
                    patterns = np.append(patterns, [new_pattern.copy()], axis=0)
                    # print(f"added_pattern {new_index}")
                base_pattern = new_pattern.copy()
    logging.info(f"{len(patterns)} unique patterns after symmetry {symmetry}")
    print(patterns.shape)
    # assert False
    # print(pattern_catalog)
    # print(patterns)
    # assert False
    return pattern_catalog, pattern_weights, patterns, pattern_grid


def find_last_patterns(pattern_grid, number_of_last_patterns):
    last_patterns = []
    for x in reversed(range(pattern_grid.shape[0])):
        for y in reversed(range(pattern_grid.shape[1])):
            current_pattern = pattern_grid[x][y]
            if not (current_pattern in last_patterns):
                last_patterns.append(current_pattern)
                if len(last_patterns) >= abs(number_of_last_patterns):
                    return last_patterns
    return last_patterns


def detect_ground(wfc_ns):
    wfc_ns.last_patterns = []
    if wfc_ns.ground != 0:
        wfc_ns.last_patterns = find_last_patterns(wfc_ns.pattern_grid, wfc_ns.ground)
    return wfc_ns


def make_pattern_catalog_with_rotation(tile_grid, pattern_width):
    the_pattern_catalog = {}
    the_pattern_weights = {}
    the_patterns = np.zeros((0, pattern_width, pattern_width), dtype=np.int64)
    number_of_rotations = 4
    for rotation in range(0, number_of_rotations):
        rotated_grid = np.rot90(tile_grid, rotation)
        flipped_grid = np.flip(rotated_grid, 0)
        for grid in [rotated_grid, flipped_grid]:
            (
                a_pattern_catalog,
                a_pattern_weights,
                a_patterns,
            ) = make_pattern_catalog_from_grid(tile_grid, pattern_width)
            the_pattern_catalog = {**the_pattern_catalog, **a_pattern_catalog}
            the_pattern_weights = {
                x: (the_pattern_weights.get(x, 0) + a_pattern_weights.get(x, 0))
                for x in (the_pattern_weights.keys() | a_pattern_weights.keys())
            }
            the_patterns = np.concatenate((the_patterns, a_patterns))
    return the_pattern_catalog, the_pattern_weights, the_patterns


def make_pattern_catalog(tile_grid, pattern_width):
    pc, pw, patterns = make_pattern_catalog_with_rotation(tile_grid, pattern_width)
    patterns = np.unique(patterns, axis=0)
    for p in range(patterns.shape[0]):
        assert np.array_equal(patterns[p], pc[p])
    return pc, pw, patterns


def make_pattern_catalog_no_rotations(tile_grid, pattern_width):
    """
    >>> make_pattern_catalog_no_rotations(test_ns.tile_grid, test_ns.pattern_width)
    ({0: array([[-8754995591521426669,                    0],
           [-8754995591521426669, -8754995591521426669]], dtype=int64), 1: array([[-8754995591521426669,                    0],
           [-8754995591521426669,                    0]], dtype=int64), 2: array([[                   0,                    0],
           [-8754995591521426669, -8754995591521426669]], dtype=int64), 3: array([[8253868773529191888,                   0],
           [                  0,                   0]], dtype=int64), 4: array([[-8754995591521426669, -8754995591521426669],
           [                   0, -8754995591521426669]], dtype=int64), 5: array([[-8754995591521426669, -8754995591521426669],
           [-8754995591521426669,                    0]], dtype=int64), 6: array([[                   0, -8754995591521426669],
           [-8754995591521426669, -8754995591521426669]], dtype=int64), 7: array([[                  0,                   0],
           [                  0, 8253868773529191888]], dtype=int64), 8: array([[                  0,                   0],
           [8253868773529191888,                   0]], dtype=int64), 9: array([[-8754995591521426669, -8754995591521426669],
           [                   0,                    0]], dtype=int64), 10: array([[                   0, -8754995591521426669],
           [                   0, -8754995591521426669]], dtype=int64), 11: array([[                  0, 8253868773529191888],
           [                  0,                   0]], dtype=int64)}, {0: 1, 1: 2, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 2, 10: 2, 11: 1}, array([[[-8754995591521426669,                    0],
            [-8754995591521426669, -8754995591521426669]],
    <BLANKLINE>
           [[-8754995591521426669,                    0],
            [-8754995591521426669,                    0]],
    <BLANKLINE>
           [[                   0,                    0],
            [-8754995591521426669, -8754995591521426669]],
    <BLANKLINE>
           [[ 8253868773529191888,                    0],
            [                   0,                    0]],
    <BLANKLINE>
           [[-8754995591521426669, -8754995591521426669],
            [                   0, -8754995591521426669]],
    <BLANKLINE>
           [[-8754995591521426669, -8754995591521426669],
            [-8754995591521426669,                    0]],
    <BLANKLINE>
           [[                   0, -8754995591521426669],
            [-8754995591521426669, -8754995591521426669]],
    <BLANKLINE>
           [[                   0,                    0],
            [                   0,  8253868773529191888]],
    <BLANKLINE>
           [[                   0,                    0],
            [ 8253868773529191888,                    0]],
    <BLANKLINE>
           [[-8754995591521426669, -8754995591521426669],
            [                   0,                    0]],
    <BLANKLINE>
           [[                   0, -8754995591521426669],
            [                   0, -8754995591521426669]],
    <BLANKLINE>
           [[                   0,  8253868773529191888],
            [                   0,                    0]]], dtype=int64))
    """
    pc, pw, patterns = make_pattern_catalog_from_grid(tile_grid, pattern_width)
    return pc, pw, patterns


# In[13]:


def version_check(minimum_ver):
    minim = [int(v) for v in minimum_ver.split(".")]
    checking = [int(v) for v in np.__version__.split(".")]
    if not checking[0] >= minim[0]:
        if not checking[1] >= minim[1]:
            if not checking[2] >= minim[2]:
                return True
    return False


def render_pattern(render_pattern, nspace):
    rp_iter = np.nditer(render_pattern, flags=["multi_index"])
    output = np.zeros(render_pattern.shape + (3,), dtype=np.uint32)
    while not rp_iter.finished:
        # Note that this truncates images with more than 3 channels down to just the channels in the output.
        # If we want to have alpha channels, we'll need a different way to handle this.
        output[rp_iter.multi_index] = np.resize(
            nspace.tile_catalog[render_pattern[rp_iter.multi_index]],
            output[rp_iter.multi_index].shape,
        )
        rp_iter.iternext()
    return output


def show_pattern_catalog(img_ns):
    s_columns = 24 // min(24, img_ns.pattern_width)
    s_rows = 1 + (int(len(img_ns.pattern_catalog)) // s_columns)
    fig = figure(figsize=(s_columns, s_rows * 1.5))
    title("Extracted Patterns")
    for i, tcode in img_ns.pattern_catalog.items():
        pat_cat = img_ns.pattern_catalog[i]
        ptr = render_pattern(pat_cat, img_ns).astype(np.uint8)
        sp = subplot(s_rows, s_columns, i + 1)
        spi = sp.imshow(ptr)
        spi.axes.xaxis.set_label_text(f"({img_ns.pattern_weights[i]})")
        sp.set_title(i)
        spi.axes.tick_params(
            labelleft=False, labelbottom=False, left=False, bottom=False
        )
        spi.axes.grid(False)
    plt.savefig(img_ns.output_filename + "_patterns.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import types
    import wfc_tiles

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
    test_ns.img = wfc_tiles.load_source_image(test_ns.img_filename)
    (
        test_ns.tile_catalog,
        test_ns.tile_grid,
        test_ns.code_list,
        test_ns.unique_tiles,
    ) = wfc_tiles.make_tile_catalog(test_ns)
    test_ns.tile_ids = {
        v: k for k, v in dict(enumerate(test_ns.unique_tiles[0])).items()
    }
    test_ns.tile_weights = {
        a: b for a, b in zip(test_ns.unique_tiles[0], test_ns.unique_tiles[1])
    }
    (
        test_ns.pattern_catalog,
        test_ns.pattern_weights,
        test_ns.patterns,
    ) = make_pattern_catalog_no_rotations(test_ns.tile_grid, test_ns.pattern_width)
    import doctest

    doctest.testmod()
