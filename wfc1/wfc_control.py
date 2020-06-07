# -*- coding: utf-8 -*-


import types
import wfc.wfc_utilities
from wfc.wfc_utilities import CoordXY, CoordRC, hash_downto, find_pattern_center
from wfc.wfc_utilities import WFC_PARTIAL_BLANK, WFC_NULL_VALUE
from wfc.wfc_tiles import (
    load_source_image,
    image_to_tiles,
    make_tile_catalog,
    show_input_to_output,
    show_extracted_tiles,
    show_false_color_tile_grid,
)
import wfc.wfc_patterns
from wfc.wfc_patterns import (
    make_pattern_catalog_no_rotations,
    show_pattern_catalog,
    make_pattern_catalog_with_symmetry,
)
from wfc.wfc_adjacency import adjacency_extraction_consistent, show_adjacencies
import wfc.wfc_solver
from wfc.wfc_solver import wfc_run
from wfc.wfc_solver import (
    wfc_init,
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

# from wfc.wfc_minizinc import mz_run

import logging

logging.basicConfig(level=logging.INFO)
wfc_logger = logging.getLogger()

import numpy as np

wfc_logger.info(f"Using numpy version {np.__version__}")

np.set_printoptions(threshold=np.inf)

import xml.etree.ElementTree as ET

import cProfile, pstats
import time

import moviepy.editor as mpy
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


import copy


def string2bool(strn):
    if isinstance(strn, bool):
        return strn
    return strn.lower() in ["true"]


def wfc_execute(WFC_VISUALIZE=False, WFC_PROFILE=False, WFC_LOGGING=False):

    solver_to_use = "default"  # "minizinc"

    wfc_stats_tracking = {
        "observations": 0,
        "propagations": 0,
        "time_start": None,
        "time_end": None,
        "choices_before_success": 0,
        "choices_per_run": [],
        "success": False,
    }
    wfc_stats_data = []
    stats_file_name = f"output/stats_{time.time()}.tsv"

    with open(stats_file_name, "a+") as stats_file:
        stats_file.write(
            "id\tname\tsuccess?\tattempts\tobservations\tpropagations\tchoices_to_solution\ttotal_observations_before_solution_in_last_restart\ttotal_choices_before_success_across_restarts\tbacktracking_total\ttime_passed\ttime_start\ttime_end\tfinal_time_end\tgenerated_size\tpattern_count\tseed\tbacktracking?\tallowed_restarts\tforce_the_use_of_all_patterns?\toutput_filename\n"
        )

    default_backtracking = False
    default_allowed_attempts = 10
    default_force_use_all_patterns = False

    xdoc = ET.ElementTree(file="samples_original.xml")
    counter = 0
    choices_before_success = 0
    for xnode in xdoc.getroot():
        counter += 1
        choices_before_success = 0
        if "#comment" == xnode.tag:
            continue

        name = xnode.get("name", "NAME")
        global hackstring
        hackstring = name
        print("< {0} ".format(name), end="")
        if "backtracking_on" == xnode.tag:
            default_backtracking = True
        if "backtracking_off" == xnode.tag:
            default_backtracking = False
        if "one_allowed_attempts" == xnode.tag:
            default_allowed_attempts = 1
        if "ten_allowed_attempts" == xnode.tag:
            default_allowed_attempts = 10
        if "force_use_all_patterns" == xnode.tag:
            default_force_use_all_patterns = True
        if "overlapping" == xnode.tag:
            choices_before_success = 0
            print("beginning...")
            print(xnode.attrib)
            current_output_file_number = 97000 + (counter * 10)
            wfc_ns = types.SimpleNamespace(
                output_path="output/",
                img_filename="samples/"
                + xnode.get("name", "NAME")
                + ".png",  # name of the input file
                output_file_number=current_output_file_number,
                operation_name=xnode.get("name", "NAME"),
                output_filename="output/"
                + xnode.get("name", "NAME")
                + "_"
                + str(current_output_file_number)
                + "_"
                + str(time.time())
                + ".png",  # name of the output file
                debug_log_filename="output/"
                + xnode.get("name", "NAME")
                + "_"
                + str(current_output_file_number)
                + "_"
                + str(time.time())
                + ".log",
                seed=11975,  # seed for random generation, can be any number
                tile_size=int(xnode.get("tile_size", 1)),  # size of tile, in pixels
                pattern_width=int(
                    xnode.get("N", 2)
                ),  # Size of the patterns we want. 2x2 is the minimum, larger scales get slower fast.
                channels=3,  # Color channels in the image (usually 3 for RGB)
                symmetry=int(xnode.get("symmetry", 8)),
                ground=int(xnode.get("ground", 0)),
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
                periodic_input=string2bool(
                    xnode.get("periodicInput", True)
                ),  # Does the input wrap?
                periodic_output=string2bool(
                    xnode.get("periodicOutput", False)
                ),  # Do we want the output to wrap?
                generated_size=(
                    int(xnode.get("width", 48)),
                    int(xnode.get("height", 48)),
                ),  # Size of the final image
                screenshots=int(
                    xnode.get("screenshots", 3)
                ),  # Number of times to run the algorithm, will produce this many distinct outputs
                iteration_limit=int(
                    xnode.get("iteration_limit", 0)
                ),  # After this many iterations, time out. 0 = never time out.
                allowed_attempts=int(
                    xnode.get("allowed_attempts", default_allowed_attempts)
                ),  # Give up after this many contradictions
                stats_tracking=wfc_stats_tracking.copy(),
                backtracking=string2bool(
                    xnode.get("backtracking", default_backtracking)
                ),
                force_use_all_patterns=default_force_use_all_patterns,
                force_fail_first_solution=False,
            )
            wfc_ns.stats_tracking["choices_before_success"] += choices_before_success
            wfc_ns.stats_tracking["time_start"] = time.time()
            pr = cProfile.Profile()
            pr.enable()
            wfc_ns = find_pattern_center(wfc_ns)
            wfc_ns = wfc.wfc_utilities.load_visualizer(wfc_ns)
            ##
            ## Load image and make tile data structures
            ##
            wfc_ns.img = load_source_image(wfc_ns.img_filename)
            wfc_ns.channels = wfc_ns.img.shape[
                -1
            ]  # detect if it uses channels other than RGB...
            wfc_ns.tiles = image_to_tiles(wfc_ns.img, wfc_ns.tile_size)
            (
                wfc_ns.tile_catalog,
                wfc_ns.tile_grid,
                wfc_ns.code_list,
                wfc_ns.unique_tiles,
            ) = make_tile_catalog(wfc_ns)
            wfc_ns.tile_ids = {
                v: k for k, v in dict(enumerate(wfc_ns.unique_tiles[0])).items()
            }
            wfc_ns.tile_weights = {
                a: b for a, b in zip(wfc_ns.unique_tiles[0], wfc_ns.unique_tiles[1])
            }

            if WFC_VISUALIZE:
                show_input_to_output(wfc_ns)
                show_extracted_tiles(wfc_ns)
                show_false_color_tile_grid(wfc_ns)

            (
                wfc_ns.pattern_catalog,
                wfc_ns.pattern_weights,
                wfc_ns.patterns,
                wfc_ns.pattern_grid,
            ) = make_pattern_catalog_with_symmetry(
                wfc_ns.tile_grid,
                wfc_ns.pattern_width,
                wfc_ns.symmetry,
                wfc_ns.periodic_input,
            )
            if WFC_VISUALIZE:
                show_pattern_catalog(wfc_ns)
            adjacency_relations = adjacency_extraction_consistent(
                wfc_ns, wfc_ns.patterns
            )
            if WFC_VISUALIZE:
                show_adjacencies(wfc_ns, adjacency_relations[:256])
            wfc_ns = wfc.wfc_patterns.detect_ground(wfc_ns)
            pr.disable()

            screenshots_collected = 0
            while screenshots_collected < wfc_ns.screenshots:
                wfc_logger.info(f"Starting solver #{screenshots_collected}")
                screenshots_collected += 1
                wfc_ns.seed += 100

                choice_before_success = 0
                # wfc_ns.stats_tracking["choices_before_success"] = 0# += choices_before_success
                wfc_ns.stats_tracking["time_start"] = time.time()
                wfc_ns.stats_tracking["final_time_end"] = None

                # update output name so each iteration has a unique filename
                output_filename = (
                    "output/"
                    + xnode.get("name", "NAME")
                    + "_"
                    + str(current_output_file_number)
                    + "_"
                    + str(time.time())
                    + "_"
                    + str(wfc_ns.seed)
                    + ".png",
                )  # name of the output file

                profile_filename = (
                    ""
                    + str(wfc_ns.output_path)
                    + "setup_"
                    + str(wfc_ns.output_file_number)
                    + "_"
                    + str(wfc_ns.seed)
                    + "_"
                    + str(time.time())
                    + "_"
                    + str(wfc_ns.seed)
                    + ".profile"
                )
                if WFC_PROFILE:
                    with open(profile_filename, "w") as profile_file:
                        ps = pstats.Stats(pr, stream=profile_file)
                        ps.sort_stats("cumtime", "ncalls")
                        ps.print_stats(20)
                solution = None

                if "minizinc" == solver_to_use:
                    attempt_count = 0
                    # while attempt_count < wfc_ns.allowed_attempts:
                    #    attempt_count += 1
                    #    solution = mz_run(wfc_ns)
                    #    solution.wfc_ns.stats_tracking["attempt_count"] = attempt_count
                    #    solution.wfc_ns.stats_tracking["choices_before_success"] += solution.wfc_ns.stats_tracking["observations"]

                else:
                    if True:
                        attempt_count = 0
                        # print("allowed attempts: " + str(wfc_ns.allowed_attempts))
                        attempt_wfc_ns = copy.deepcopy(wfc_ns)
                        attempt_wfc_ns.stats_tracking["time_start"] = time.time()
                        attempt_wfc_ns.stats_tracking["choices_before_success"] = 0
                        attempt_wfc_ns.stats_tracking[
                            "total_observations_before_success"
                        ] = 0
                        wfc.wfc_solver.reset_backtracking_count()  # reset the count of how many times we've backtracked, because multiple attempts are handled here instead of there
                        while attempt_count < wfc_ns.allowed_attempts:
                            attempt_count += 1
                            print(attempt_count, end=" ")
                            attempt_wfc_ns.seed += 7  # change seed for each attempt...
                            solution = wfc_run(
                                attempt_wfc_ns,
                                visualize=WFC_VISUALIZE,
                                logging=WFC_LOGGING,
                            )
                            solution.wfc_ns.stats_tracking[
                                "attempt_count"
                            ] = attempt_count
                            solution.wfc_ns.stats_tracking[
                                "choices_before_success"
                            ] += solution.wfc_ns.stats_tracking["observations"]
                            attempt_wfc_ns.stats_tracking[
                                "total_observations_before_success"
                            ] += solution.wfc_ns.stats_tracking["total_observations"]
                            wfc_logger.info(
                                "result: {} is {}".format(
                                    attempt_count, solution.result
                                )
                            )
                            if solution.result == -2:
                                attempt_count = wfc_ns.allowed_attempts
                                solution.wfc_ns.stats_tracking["time_end"] = time.time()
                            wfc_stats_data.append(solution.wfc_ns.stats_tracking.copy())
                    solution.wfc_ns.stats_tracking["final_time_end"] = time.time()
                    print("tracking choices before success...")
                    choices_before_success = solution.wfc_ns.stats_tracking[
                        "choices_before_success"
                    ]
                    time_passed = None
                    if None != solution.wfc_ns.stats_tracking["time_end"]:
                        time_passed = (
                            solution.wfc_ns.stats_tracking["time_end"]
                            - solution.wfc_ns.stats_tracking["time_start"]
                        )
                    else:
                        if None != solution.wfc_ns.stats_tracking["final_time_end"]:
                            time_passed = (
                                solution.wfc_ns.stats_tracking["final_time_end"]
                                - solution.wfc_ns.stats_tracking["time_start"]
                            )

                    print("...finished calculating time passed")
                    # print(wfc_stats_data)
                    print("writing stats...", end="")

                    with open(stats_file_name, "a+") as stats_file:
                        stats_file.write(
                            f"{solution.wfc_ns.output_file_number}\t{solution.wfc_ns.operation_name}\t{solution.wfc_ns.stats_tracking['success']}\t{solution.wfc_ns.stats_tracking['attempt_count']}\t{solution.wfc_ns.stats_tracking['observations']}\t{solution.wfc_ns.stats_tracking['propagations']}\t{solution.wfc_ns.stats_tracking['choices_before_success']}\t{solution.wfc_ns.stats_tracking['total_observations']}\t{attempt_wfc_ns.stats_tracking['total_observations_before_success']}\t{solution.backtracking_total}\t{time_passed}\t{solution.wfc_ns.stats_tracking['time_start']}\t{solution.wfc_ns.stats_tracking['time_end']}\t{solution.wfc_ns.stats_tracking['final_time_end']}\t{solution.wfc_ns.generated_size}\t{len(solution.wfc_ns.pattern_weights.keys())}\t{solution.wfc_ns.seed}\t{solution.wfc_ns.backtracking}\t{solution.wfc_ns.allowed_attempts}\t{solution.wfc_ns.force_use_all_patterns}\t{solution.wfc_ns.output_filename}\n"
                        )
                    print("done")

                if WFC_VISUALIZE:
                    print("visualize")
                    if None == solution:
                        print("n u l l")
                    # print(solution)
                    print(1)
                    solution_vis = wfc.wfc_solver.render_recorded_visualization(
                        solution.recorded_vis
                    )
                    # print(solution)
                    print(2)

                    video_fn = f"{solution.wfc_ns.output_path}/crystal_example_{solution.wfc_ns.output_file_number}_{time.time()}.mp4"
                    wfc_logger.info("*****************************")
                    wfc_logger.warning(video_fn)
                    print(
                        f"solver recording stack len - {len(solution_vis.solver_recording_stack)}"
                    )
                    print(solution_vis.solver_recording_stack[0].shape)
                    if len(solution_vis.solver_recording_stack) > 0:
                        wfc_logger.info(solution_vis.solver_recording_stack[0].shape)
                        writer = FFMPEG_VideoWriter(
                            video_fn,
                            [
                                solution_vis.solver_recording_stack[0].shape[0],
                                solution_vis.solver_recording_stack[0].shape[1],
                            ],
                            12.0,
                        )
                        for img_data in solution_vis.solver_recording_stack:
                            writer.write_frame(img_data)
                        print("!", end="")
                        writer.close()
                        mpy.ipython_display(video_fn, height=700)
                print("recording done")
                if WFC_VISUALIZE:
                    solution = wfc_partial_output(solution)
                    show_rendered_patterns(solution, True)
                print("render to output")
                render_patterns_to_output(solution, True, False)
                print("completed")
                print("\n{0} >".format(name))

        elif "simpletiled" == xnode.tag:
            print("> ", end="\n")
            continue
        else:
            continue
