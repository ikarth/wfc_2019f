"Visualize the patterns into tiles and so on."
from __future__ import annotations

import logging
import math
import pathlib
import itertools
from typing import Dict, Tuple
import imageio  # type: ignore
import matplotlib  # type: ignore
import struct
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from numpy.typing import NDArray
from .wfc_patterns import pattern_grid_to_tiles

logger = logging.getLogger(__name__)

## Helper functions
RGB_CHANNELS = 3


def rgb_to_int(rgb_in):
    """"Takes RGB triple, returns integer representation."""
    return struct.unpack(
        "I", struct.pack("<" + "B" * 4, *(rgb_in + [0] * (4 - len(rgb_in))))
    )[0]


def int_to_rgb(val):
    """Convert hashed int to RGB values"""
    return [x for x in val.to_bytes(RGB_CHANNELS, "little")]


WFC_PARTIAL_BLANK = np.nan


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
                    pixel = tile_catalog[tile][u, v]
            new_img[u, v] = pixel
    return new_img


def argmax_unique(arr, axis):
    """Return a mask so that we can exclude the nonunique maximums, i.e. the nodes that aren't completely resolved"""
    arrm = np.argmax(arr, axis)
    arrs = np.sum(arr, axis)
    nonunique_mask = np.ma.make_mask((arrs == 1) is False)
    uni_argmax = np.ma.masked_array(arrm, mask=nonunique_mask, fill_value=-1)
    return uni_argmax, nonunique_mask


def make_solver_loggers(filename, stats={}):
    counter_choices = 0
    counter_wave = 0
    counter_backtracks = 0
    counter_propagate = 0

    def choice_count(pattern, i, j, wave=None):
        nonlocal counter_choices
        counter_choices += 1

    def wave_count(wave):
        nonlocal counter_wave
        counter_wave += 1

    def backtrack_count() -> None:
        nonlocal counter_backtracks
        counter_backtracks += 1

    def propagate_count(wave):
        nonlocal counter_propagate
        counter_propagate += 1

    def final_count(wave):
        logger.info(
            f"{filename}: choices: {counter_choices}, wave:{counter_wave}, backtracks: {counter_backtracks}, propagations: {counter_propagate}"
        )
        stats.update(
            {
                "choices": counter_choices,
                "wave": counter_wave,
                "backtracks": counter_backtracks,
                "propagations": counter_propagate,
            }
        )
        return stats

    def report_count():
        stats.update(
            {
                "choices": counter_choices,
                "wave": counter_wave,
                "backtracks": counter_backtracks,
                "propagations": counter_propagate,
            }
        )
        return stats

    return (
        choice_count,
        wave_count,
        backtrack_count,
        propagate_count,
        final_count,
        report_count,
    )


def make_solver_visualizers(
    filename: str,
    wave: NDArray[np.bool_],
    decode_patterns=None,
    pattern_catalog=None,
    tile_catalog=None,
    tile_size=[1, 1],
):
    """Construct visualizers for displaying the intermediate solver status"""
    logger.debug(wave.shape)
    pattern_total_count = wave.shape[0]
    resolution_order = np.full(
        wave.shape[1:], np.nan
    )  # pattern_wave = when was this resolved?
    backtracking_order = np.full(
        wave.shape[1:], np.nan
    )  # on which iternation was this resolved?
    pattern_solution = np.full(wave.shape[1:], np.nan)  # what is the resolved result?
    resolution_method = np.zeros(
        wave.shape[1:]
    )  # did we set this via observation or propagation?
    choice_count = 0
    vis_count = 0
    backtracking_count = 0
    max_choices = math.floor((wave.shape[1] * wave.shape[2]) / 3)
    output_individual_visualizations = False

    tile_wave = np.zeros(wave.shape, dtype=np.int64)
    for i in range(wave.shape[0]):
        local_solution_as_ids = np.full(wave.shape[1:], decode_patterns[i])
        local_solution_tile_grid = pattern_grid_to_tiles(
            local_solution_as_ids, pattern_catalog
        )
        tile_wave[i] = local_solution_tile_grid

    def choice_vis(pattern, i, j, wave=None):
        nonlocal choice_count
        nonlocal resolution_order
        nonlocal resolution_method
        choice_count += 1
        resolution_order[i][j] = choice_count
        pattern_solution[i][j] = pattern
        resolution_method[i][j] = 2
        if output_individual_visualizations:
            figure_solver_data(
                f"visualization/{filename}_choice_{choice_count}.png",
                "order of resolution",
                resolution_order,
                0,
                max_choices,
                "gist_ncar",
            )
            figure_solver_data(
                f"visualization/{filename}_solution_{choice_count}.png",
                "chosen pattern",
                pattern_solution,
                0,
                pattern_total_count,
                "viridis",
            )
            figure_solver_data(
                f"visualization/{filename}_resolution_{choice_count}.png",
                "resolution method",
                resolution_method,
                0,
                2,
                "inferno",
            )
        if wave:
            _assigned_patterns, nonunique_mask = argmax_unique(wave, 0)
            resolved_by_propagation = (
                np.ma.mask_or(nonunique_mask, resolution_method != 0) == 0
            )
            resolution_method[resolved_by_propagation] = 1
            resolution_order[resolved_by_propagation] = choice_count
            if output_individual_visualizations:
                figure_solver_data(
                    f"visualization/{filename}_wave_{choice_count}.png",
                    "patterns remaining",
                    np.count_nonzero(wave > 0, axis=0),
                    0,
                    wave.shape[0],
                    "plasma",
                )

    def wave_vis(wave):
        nonlocal vis_count
        nonlocal resolution_method
        nonlocal resolution_order
        vis_count += 1
        pattern_left_count = np.count_nonzero(wave > 0, axis=0)
        # assigned_patterns, nonunique_mask = argmax_unique(wave, 0)
        resolved_by_propagation = (
            np.ma.mask_or(pattern_left_count > 1, resolution_method != 0) != 1
        )
        # logger.debug(resolved_by_propagation)
        resolution_method[resolved_by_propagation] = 1
        resolution_order[resolved_by_propagation] = choice_count
        backtracking_order[resolved_by_propagation] = backtracking_count
        if output_individual_visualizations:
            figure_wave_patterns(filename, pattern_left_count, pattern_total_count)
            figure_solver_data(
                f"visualization/{filename}_wave_patterns_{choice_count}.png",
                "patterns remaining",
                pattern_left_count,
                0,
                pattern_total_count,
                "magma",
            )
        if decode_patterns and pattern_catalog and tile_catalog:
            solution_as_ids = np.vectorize(lambda x: decode_patterns[x])(
                np.argmax(wave, 0)
            )
            solution_tile_grid = pattern_grid_to_tiles(solution_as_ids, pattern_catalog)
            if output_individual_visualizations:
                figure_solver_data(
                    f"visualization/{filename}_tiles_assigned_{choice_count}.png",
                    "tiles assigned",
                    solution_tile_grid,
                    0,
                    pattern_total_count,
                    "plasma",
                )
            img = tile_grid_to_image(solution_tile_grid.T, tile_catalog, tile_size)

            masked_tile_wave: np.ma.MaskedArray = np.ma.MaskedArray(
                data=tile_wave, mask=(wave == False), dtype=np.int64
            )
            masked_img = tile_grid_to_average(
                np.transpose(masked_tile_wave, (0, 2, 1)), tile_catalog, tile_size
            )

            if output_individual_visualizations:
                figure_solver_image(
                    f"visualization/{filename}_solution_partial_{choice_count}.png",
                    "solved_tiles",
                    img.astype(np.uint8),
                )
                imageio.imwrite(
                    f"visualization/{filename}_solution_partial_img_{choice_count}.png",
                    img.astype(np.uint8),
                )
            fig_list = [
                # {"title": "resolved by propagation", "data": resolved_by_propagation.T, "vmin": 0, "vmax": 2, "cmap": "inferno", "datatype":"figure"},
                {
                    "title": "order of resolution",
                    "data": resolution_order.T,
                    "vmin": 0,
                    "vmax": max_choices / 4,
                    "cmap": "hsv",
                    "datatype": "figure",
                },
                {
                    "title": "chosen pattern",
                    "data": pattern_solution.T,
                    "vmin": 0,
                    "vmax": pattern_total_count,
                    "cmap": "viridis",
                    "datatype": "figure",
                },
                {
                    "title": "resolution method",
                    "data": resolution_method.T,
                    "vmin": 0,
                    "vmax": 2,
                    "cmap": "magma",
                    "datatype": "figure",
                },
                {
                    "title": "patterns remaining",
                    "data": pattern_left_count.T,
                    "vmin": 0,
                    "vmax": pattern_total_count,
                    "cmap": "viridis",
                    "datatype": "figure",
                },
                {
                    "title": "tiles assigned",
                    "data": solution_tile_grid.T,
                    "vmin": None,
                    "vmax": None,
                    "cmap": "prism",
                    "datatype": "figure",
                },
                {
                    "title": "solved tiles",
                    "data": masked_img.astype(np.uint8),
                    "datatype": "image",
                },
            ]
            figure_unified(
                "Solver Readout",
                f"visualization/{filename}_readout_{choice_count:03}_{vis_count:03}.png",
                fig_list,
            )

    def backtrack_vis() -> None:
        nonlocal vis_count
        nonlocal pattern_solution
        nonlocal backtracking_count
        backtracking_count += 1
        vis_count += 1
        pattern_solution = np.full(wave.shape[1:], -1)

    return choice_vis, wave_vis, backtrack_vis, None, wave_vis, None


def figure_unified(figure_name_overall, filename, data):
    matfig, axs = plt.subplots(
        1, len(data), sharey="row", gridspec_kw={"hspace": 0, "wspace": 0}
    )

    for idx, _data_obj in enumerate(data):
        if "image" == data[idx]["datatype"]:
            axs[idx].imshow(data[idx]["data"], interpolation="nearest")
        else:
            axs[idx].matshow(
                data[idx]["data"],
                vmin=data[idx]["vmin"],
                vmax=data[idx]["vmax"],
                cmap=data[idx]["cmap"],
            )
        axs[idx].get_xaxis().set_visible(False)
        axs[idx].get_yaxis().set_visible(False)
        axs[idx].label_outer()

    plt.savefig(filename, bbox_inches="tight", pad_inches=0, dpi=600)
    plt.close(fig=matfig)
    plt.close("all")


vis_count = 0


def visualize_solver(wave):
    pattern_left_count = np.count_nonzero(wave > 0, axis=0)
    pattern_total_count = wave.shape[0]
    figure_wave_patterns(pattern_left_count, pattern_total_count)


def make_figure_solver_image(plot_title, img):
    visfig = plt.figure(figsize=(4, 4), edgecolor="k", frameon=True)
    plt.imshow(img, interpolation="nearest")
    plt.title(plot_title)
    plt.grid(None)
    plt.grid(None)
    an_ax = plt.gca()
    an_ax.get_xaxis().set_visible(False)
    an_ax.get_yaxis().set_visible(False)
    return visfig


def figure_solver_image(filename, plot_title, img):
    visfig = make_figure_solver_image(plot_title, img)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close(fig=visfig)
    plt.close("all")


def make_figure_solver_data(plot_title, data, min_count, max_count, cmap_name):
    visfig = plt.figure(figsize=(4, 4), edgecolor="k", frameon=True)
    plt.title(plot_title)
    plt.matshow(data, vmin=min_count, vmax=max_count, cmap=cmap_name)
    plt.grid(None)
    plt.grid(None)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return visfig


def figure_solver_data(filename, plot_title, data, min_count, max_count, cmap_name):
    visfig = make_figure_solver_data(plot_title, data, min_count, max_count, cmap_name)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close(fig=visfig)
    plt.close("all")


def figure_wave_patterns(filename, pattern_left_count, max_count):
    global vis_count
    vis_count += 1
    visfig = plt.figure(figsize=(4, 4), edgecolor="k", frameon=True)

    plt.title("wave")
    plt.matshow(pattern_left_count, vmin=0, vmax=max_count, cmap="plasma")
    plt.grid(None)

    plt.grid(None)
    plt.savefig(f"{filename}_wave_patterns_{vis_count}.png")
    plt.close(fig=visfig)


def tile_grid_to_average(
    tile_grid: np.ma.MaskedArray,
    tile_catalog: Dict[int, NDArray[np.int64]],
    tile_size: Tuple[int, int],
    color_channels: int = 3,
) -> NDArray[np.int64]:
    """
  Takes a masked array of tile grid stacks and transforms it into an image, taking
  the average colors of the tiles in tile_catalog.
  """
    new_img = np.zeros(
        (
            tile_grid.shape[1] * tile_size[0],
            tile_grid.shape[2] * tile_size[1],
            color_channels,
        ),
        dtype=np.int64,
    )
    for i in range(tile_grid.shape[1]):
        for j in range(tile_grid.shape[2]):
            tile_stack = tile_grid[:, i, j]
            for u in range(tile_size[0]):
                for v in range(tile_size[1]):
                    pixel = [200, 0, 200]
                    pixel_list = np.array(
                        [
                            tile_catalog[t][u, v]
                            for t in tile_stack[tile_stack.mask == False]
                        ],
                        dtype=np.int64,
                    )
                    pixel = np.mean(pixel_list, axis=0)
                    # TODO: will need to change if using an image with more than 3 channels
                    new_img[(i * tile_size[0]) + u, (j * tile_size[1]) + v] = np.resize(
                        pixel,
                        new_img[(i * tile_size[0]) + u, (j * tile_size[1]) + v].shape,
                    )
    return new_img


def tile_grid_to_image(
    tile_grid: NDArray[np.int64],
    tile_catalog: Dict[int, NDArray[np.integer]],
    tile_size: Tuple[int, int],
    visualize: bool = False,
    partial: bool = False,
    color_channels: int = 3,
) -> NDArray[np.integer]:
    """
    Takes a tile_grid and transforms it into an image, using the information
    in tile_catalog. We use tile_size to figure out the size the new image
    should be, and visualize for displaying partial tile patterns.
    """
    tile_dtype = next(iter(tile_catalog.values())).dtype
    new_img = np.zeros(
        (
            tile_grid.shape[0] * tile_size[0],
            tile_grid.shape[1] * tile_size[1],
            color_channels,
        ),
        dtype=tile_dtype,
    )
    if partial and (len(tile_grid.shape)) > 2:
        # TODO: implement rendering partially completed solution
        # Call tile_grid_to_average() instead.
        assert False
    else:
        for i in range(tile_grid.shape[0]):
            for j in range(tile_grid.shape[1]):
                tile = tile_grid[i, j]
                for u in range(tile_size[0]):
                    for v in range(tile_size[1]):
                        pixel = [200, 0, 200]
                        ## If we want to display a partial pattern, it is helpful to
                        ## be able to show empty cells. Therefore, in visualize mode,
                        ## we use -1 as a magic number for a non-existant tile.
                        if visualize and ((-1 == tile) or (-2 == tile)):
                            if -1 == tile:
                                if 0 == (i + j) % 2:
                                    pixel = [255, 0, 255]
                            if -2 == tile:
                                pixel = [0, 255, 255]
                        else:
                            pixel = tile_catalog[tile][u, v]
                        # TODO: will need to change if using an image with more than 3 channels
                        new_img[
                            (i * tile_size[0]) + u, (j * tile_size[1]) + v
                        ] = np.resize(
                            pixel,
                            new_img[
                                (i * tile_size[0]) + u, (j * tile_size[1]) + v
                            ].shape,
                        )
    return new_img


def figure_list_of_tiles(unique_tiles, tile_catalog, output_filename="list_of_tiles"):
    plt.figure(figsize=(4, 4), edgecolor="k", frameon=True)
    plt.title("Extracted Tiles")
    s = math.ceil(math.sqrt(len(unique_tiles))) + 1
    for i, tcode in enumerate(unique_tiles[0]):
        sp = plt.subplot(s, s, i + 1).imshow(tile_catalog[tcode])
        sp.axes.tick_params(labelleft=False, labelbottom=False, length=0)
        plt.title(f"{i}\n{tcode}", fontsize=10)
        sp.axes.grid(False)
    fp = pathlib.Path(output_filename + ".pdf")
    plt.savefig(fp, bbox_inches="tight")
    plt.close()


def figure_false_color_tile_grid(tile_grid, output_filename="./false_color_tiles"):
    figure_plot = plt.matshow(
        tile_grid,
        cmap="gist_ncar",
        extent=(0, tile_grid.shape[1], tile_grid.shape[0], 0),
    )
    plt.title("False Color Map of Tiles in Input Image")
    figure_plot.axes.grid(None)
    plt.savefig(output_filename + ".png", bbox_inches="tight")
    plt.close()


def figure_tile_grid(tile_grid, tile_catalog, tile_size):
    tile_grid_to_image(tile_grid, tile_catalog, tile_size)


def render_pattern(render_pattern, tile_catalog):
    """Turn a pattern into an image"""
    rp_iter = np.nditer(render_pattern, flags=["multi_index"])
    output = np.zeros(render_pattern.shape + (3,), dtype=np.uint32)
    while not rp_iter.finished:
        # Note that this truncates images with more than 3 channels down to just the channels in the output.
        # If we want to have alpha channels, we'll need a different way to handle this.
        output[rp_iter.multi_index] = np.resize(
            tile_catalog[render_pattern[rp_iter.multi_index]],
            output[rp_iter.multi_index].shape,
        )
        rp_iter.iternext()
    return output


def figure_pattern_catalog(
    pattern_catalog,
    tile_catalog,
    pattern_weights,
    pattern_width,
    output_filename="pattern_catalog",
):
    s_columns = 24 // min(24, pattern_width)
    s_rows = 1 + (int(len(pattern_catalog)) // s_columns)
    _fig = plt.figure(figsize=(s_columns, s_rows * 1.5))
    plt.title("Extracted Patterns")
    counter = 0
    for i, _tcode in pattern_catalog.items():
        pat_cat = pattern_catalog[i]
        ptr = render_pattern(pat_cat, tile_catalog).astype(np.uint8)
        sp = plt.subplot(s_rows, s_columns, counter + 1)
        spi = sp.imshow(ptr)
        spi.axes.xaxis.set_label_text(f"({pattern_weights[i]})")
        sp.set_title(f"{counter}\n{i}", fontsize=3)
        spi.axes.tick_params(
            labelleft=False, labelbottom=False, left=False, bottom=False
        )
        spi.axes.grid(False)
        counter += 1
    plt.savefig(output_filename + "_patterns.pdf", bbox_inches="tight")
    plt.close()


def render_tiles_to_output(
    tile_grid: NDArray[np.int64],
    tile_catalog: Dict[int, NDArray[np.integer]],
    tile_size: Tuple[int, int],
    output_filename: str,
) -> None:
    img = tile_grid_to_image(tile_grid.T, tile_catalog, tile_size)
    imageio.imwrite(output_filename, img.astype(np.uint8))


def blit(destination, sprite, upper_left, layer=False, check=False):
    """
    Blits one multidimensional array into another numpy array.
    """
    lower_right = [
        ((a + b) if ((a + b) < c) else c)
        for a, b, c in zip(upper_left, sprite.shape, destination.shape)
    ]
    if min(lower_right) < 0:
        return

    for i_index, i in enumerate(range(upper_left[0], lower_right[0])):
        for j_index, j in enumerate(range(upper_left[1], lower_right[1])):
            if (i >= 0) and (j >= 0):
                if len(destination.shape) > 2:
                    destination[i, j, layer] = sprite[i_index, j_index]
                else:
                    if check:
                        if (
                            (destination[i, j] == sprite[i_index, j_index])
                            or (destination[i, j] == -1)
                            or {sprite[i_index, j_index] == -1}
                        ):
                            destination[i, j] = sprite[i_index, j_index]
                        else:
                            logger.error(
                                "mismatch: destination[{i},{j}] = {destination[i, j]}, sprite[{i_index}, {j_index}] = {sprite[i_index, j_index]}"
                            )
                    else:
                        destination[i, j] = sprite[i_index, j_index]
    return destination


class InvalidAdjacency(Exception):
    """The combination of patterns and offsets results in pattern combinations that don't match."""

    pass


def validate_adjacency(
    pattern_a, pattern_b, preview_size, upper_left_of_center, adj_rel
):
    preview_adj_a_first = np.full((preview_size, preview_size), -1, dtype=np.int64)
    preview_adj_b_first = np.full((preview_size, preview_size), -1, dtype=np.int64)
    blit(
        preview_adj_b_first,
        pattern_b,
        (
            upper_left_of_center[1] + adj_rel[0][1],
            upper_left_of_center[0] + adj_rel[0][0],
        ),
        check=True,
    )
    blit(preview_adj_b_first, pattern_a, upper_left_of_center, check=True)

    blit(preview_adj_a_first, pattern_a, upper_left_of_center, check=True)
    blit(
        preview_adj_a_first,
        pattern_b,
        (
            upper_left_of_center[1] + adj_rel[0][1],
            upper_left_of_center[0] + adj_rel[0][0],
        ),
        check=True,
    )
    if not np.array_equiv(preview_adj_a_first, preview_adj_b_first):
        logger.debug(adj_rel)
        logger.debug(pattern_a)
        logger.debug(pattern_b)
        logger.debug(preview_adj_a_first)
        logger.debug(preview_adj_b_first)
        raise InvalidAdjacency


def figure_adjacencies(
    adjacency_relations_list,
    adjacency_directions,
    tile_catalog,
    patterns,
    pattern_width,
    tile_size,
    output_filename="adjacency",
    render_b_first=False,
):
    #    try:
    adjacency_directions_list = list(dict(adjacency_directions).values())
    _figadj = plt.figure(
        figsize=(12, 1 + len(adjacency_relations_list[:64])), edgecolor="b"
    )
    plt.title("Adjacencies")
    max_offset = max(
        [abs(x) for x in list(itertools.chain.from_iterable(adjacency_directions_list))]
    )

    for i, adj_rel in enumerate(adjacency_relations_list[:64]):
        preview_size = pattern_width + max_offset * 2
        preview_adj = np.full((preview_size, preview_size), -1, dtype=np.int64)
        upper_left_of_center = [max_offset, max_offset]

        pattern_a = patterns[adj_rel[1]]
        pattern_b = patterns[adj_rel[2]]
        validate_adjacency(
            pattern_a, pattern_b, preview_size, upper_left_of_center, adj_rel
        )
        if render_b_first:
            blit(
                preview_adj,
                pattern_b,
                (
                    upper_left_of_center[1] + adj_rel[0][1],
                    upper_left_of_center[0] + adj_rel[0][0],
                ),
                check=True,
            )
            blit(preview_adj, pattern_a, upper_left_of_center, check=True)
        else:
            blit(preview_adj, pattern_a, upper_left_of_center, check=True)
            blit(
                preview_adj,
                pattern_b,
                (
                    upper_left_of_center[1] + adj_rel[0][1],
                    upper_left_of_center[0] + adj_rel[0][0],
                ),
                check=True,
            )

        ptr = tile_grid_to_image(
            preview_adj, tile_catalog, tile_size, visualize=True
        ).astype(np.uint8)

        subp = plt.subplot(math.ceil(len(adjacency_relations_list[:64]) / 4), 4, i + 1)
        spi = subp.imshow(ptr)
        spi.axes.tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )
        plt.title(
            f"{i}:\n({adj_rel[1]} +\n{adj_rel[2]})\n by {adj_rel[0]}", fontsize=10
        )

        indicator_rect = matplotlib.patches.Rectangle(
            (upper_left_of_center[1] - 0.51, upper_left_of_center[0] - 0.51),
            pattern_width,
            pattern_width,
            Fill=False,
            edgecolor="b",
            linewidth=3.0,
            linestyle=":",
        )

        spi.axes.add_artist(indicator_rect)
        spi.axes.grid(False)
    plt.savefig(output_filename + "_adjacency.pdf", bbox_inches="tight")
    plt.close()


#    except ValueError as e:
#        logger.exception(e)
