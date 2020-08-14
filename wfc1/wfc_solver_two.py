#

import types
import collections
import logging
import math
import pdb
import numpy as np


from wfc.wfc_adjacency import adjacency_extraction_consistent

from wfc.wfc_utilities import WFC_PARTIAL_BLANK, WFC_NULL_VALUE

# import matplotlib.pyplot
# from matplotlib.pyplot import figure, subplot, subplots, title, matshow
# from wfc.wfc_patterns import render_pattern
# from wfc.wfc_adjacency import blit
# from wfc.wfc_tiles import tiles_to_images
import wfc.wfc_utilities
from wfc.wfc_utilities import CoordXY, CoordRC, hash_downto, find_pattern_center

# import random
# import copy
# import time

# import imageio

logging.basicConfig(level=logging.INFO)
WFC_LOGGER = logging.getLogger()


WFC_FINISHED = -2
WFC_FAILURE = -1
WFC_TIMEDOUT = -3
WFC_FAKE_FAILURE = -6


def weight_log(val):
    """Return the log of the weight value, used in calculating updated weights."""
    return val * math.log(val)


def wfc_init(prestate):
    """
    Initialize the WFC solver, returning the fixed and mutable data structures needed.
    """
    prestate.adjacency_directions_rc = {
        i: CoordRC(a.y, a.x) for i, a in prestate.adjacency_directions.items()
    }  #
    prestate = wfc.wfc_utilities.find_pattern_center(prestate)
    parameters = types.SimpleNamespace(wfc_ns=prestate)

    state = types.SimpleNamespace()
    state.result = None

    parameters.heuristic = (
        0  # TODO: Implement control code to choose between heuristics
    )

    parameters.adjacency_relations = adjacency_extraction_consistent(
        parameters.wfc_ns, parameters.wfc_ns.patterns
    )
    parameters.patterns = np.array(list(parameters.wfc_ns.pattern_catalog.keys()))
    parameters.pattern_translations = list(parameters.wfc_ns.pattern_catalog.values())
    parameters.number_of_patterns = parameters.patterns.size
    parameters.number_of_directions = len(parameters.wfc_ns.adjacency_directions)

    # The Propagator is a data structure that holds the adjacency information
    # for the patterns, i.e. given a direction, which patterns are allowed to
    # be placed next to the pattern that we're currently concerned with.
    # This won't change over the course of using the solver, so the important
    # thing here is fast lookup.
    parameters.propagator_matrix = np.zeros(
        (
            parameters.number_of_directions,
            parameters.number_of_patterns,
            parameters.number_of_patterns,
        ),
        dtype=np.bool_,
    )
    for direction, pattern_one, pattern_two in parameters.adjacency_relations:
        parameters.propagator_matrix[(direction, pattern_one, pattern_two)] = True

    output = types.SimpleNamespace()

    # The Wave Table is the boolean expression table of which patterns are allowed
    # in which cells of the solution we are calculating.
    parameters.rows = parameters.wfc_ns.generated_size[0]
    parameters.columns = parameters.wfc_ns.generated_size[1]

    output.solving_time = np.full(
        (parameters.rows, parameters.columns), 0, dtype=np.int32
    )
    output.propagation_time = np.full(
        (parameters.rows, parameters.columns), 0, dtype=np.int32
    )

    parameters.wave_shape = [
        parameters.rows,
        parameters.columns,
        parameters.number_of_patterns,
    ]
    state.wave_table = np.full(parameters.wave_shape, True, dtype=np.bool_)

    # The compatible_count is a running count of the number of patterns that
    # are still allowed to be next to this cell in a particular direction.
    compatible_shape = [
        parameters.rows,
        parameters.columns,
        parameters.number_of_patterns,
        parameters.number_of_directions,
    ]

    WFC_LOGGER.debug(f"compatible shape:{compatible_shape}")
    state.compatible_count = np.full(
        compatible_shape, parameters.number_of_patterns, dtype=np.int16
    )  # assumes that there are less than 65536 patterns

    # The weights are how we manage the probabilities when we choose the next
    # pattern to place. Rather than recalculating them from scratch each time,
    # these let us incrementally update their values.
    state.weights = np.array(list(parameters.wfc_ns.pattern_weights.values()))
    state.weight_log_weights = np.vectorize(weight_log)(state.weights)
    state.sum_of_weights = np.sum(state.weights)

    state.sum_of_weight_log_weights = np.sum(state.weight_log_weights)
    state.starting_entropy = math.log(state.sum_of_weights) - (
        state.sum_of_weight_log_weights / state.sum_of_weights
    )

    state.entropies = np.zeros([parameters.rows, parameters.columns], dtype=np.float64)
    state.sums_of_weights = np.zeros(
        [parameters.rows, parameters.columns], dtype=np.float64
    )

    # Instead of updating all of the cells for every propagation, we use a queue
    # that marks the dirty tiles to update.
    state.observation_stack = collections.deque()

    output.output_grid = np.full(
        [parameters.rows, parameters.columns], WFC_NULL_VALUE, dtype=np.int64
    )
    output.partial_output_grid = np.full(
        [parameters.rows, parameters.columns, parameters.number_of_patterns],
        -9,
        dtype=np.int64,
    )

    output.current_iteration_count_observation = 0
    output.current_iteration_count_propagation = 0
    output.current_iteration_count_last_touch = 0
    output.current_iteration_count_crystal = 0
    output.solving_time = np.full(
        (parameters.rows, parameters.columns), 0, dtype=np.int32
    )
    output.ones_time = np.full((parameters.rows, parameters.columns), 0, dtype=np.int32)
    output.propagation_time = np.full(
        (parameters.rows, parameters.columns), 0, dtype=np.int32
    )
    output.touch_time = np.full(
        (parameters.rows, parameters.columns), 0, dtype=np.int32
    )
    output.crystal_time = np.full(
        (parameters.rows, parameters.columns), 0, dtype=np.int32
    )
    output.method_time = np.full(
        (parameters.rows, parameters.columns), 0, dtype=np.int32
    )
    output.choices_recording = np.full(
        (parameters.rows, parameters.columns), 0, dtype=np.float32
    )

    output.stats_tracking = prestate.stats_tracking.copy()

    return parameters, state, output


def wfc_clear(parameters, state, output):
    """Given an initialized WFC solver state, clear it out for the beginning for the solving."""
    # Crystal solving time matrix
    output.current_iteration_count_observation = 0
    output.current_iteration_count_propagation = 0
    output.current_iteration_count_last_touch = 0
    output.current_iteration_count_crystal = 0

    output.solving_time = np.full(
        (parameters.rows, parameters.columns), 0, dtype=np.int32
    )
    output.ones_time = np.full((parameters.rows, parameters.columns), 0, dtype=np.int32)
    output.propagation_time = np.full(
        (parameters.rows, parameters.columns), 0, dtype=np.int32
    )
    output.touch_time = np.full(
        (parameters.rows, parameters.columns), 0, dtype=np.int32
    )
    output.crystal_time = np.full(
        (parameters.rows, parameters.columns), 0, dtype=np.int32
    )
    output.method_time = np.full(
        (parameters.rows, parameters.columns), 0, dtype=np.int32
    )
    output.choices_recording = np.full(
        (parameters.rows, parameters.columns), 0, dtype=np.float32
    )
    output.stats_tracking["total_observations"] = 0

    state.wave_table = np.full(parameters.wave_shape, True, dtype=np.bool_)

    compatible_shape = [
        parameters.rows,
        parameters.columns,
        parameters.number_of_patterns,
        parameters.number_of_directions,
    ]

    # Initialize the compatible count from the propagation matrix. This sets the
    # maximum domain of possible neighbors for each cell node.

    def prop_compat(p, d):
        return sum(parameters.propagator_matrix[(d + 2) % 4][p])

    def comp_count(r, c, p, d):
        return pattern_compatible_count[p][d]

    pcomp = np.vectorize(prop_compat)
    ccount = np.vectorize(comp_count)
    pattern_compatible_count = np.fromfunction(
        pcomp,
        (parameters.number_of_patterns, parameters.number_of_directions),
        dtype=np.int16,
    )
    state.compatible_count = np.fromfunction(
        ccount,
        (
            parameters.rows,
            parameters.columns,
            parameters.number_of_patterns,
            parameters.number_of_directions,
        ),
        dtype=np.int16,
    )

    # Likewise, set the weights to their maximum values
    state.sums_of_weights = np.full(
        [parameters.rows, parameters.columns], state.sum_of_weights, dtype=np.float64
    )
    state.sums_of_weight_log_weights = np.full(
        [parameters.rows, parameters.columns],
        state.sum_of_weight_log_weights,
        dtype=np.float64,
    )
    state.entropies = np.full(
        [parameters.rows, parameters.columns], state.starting_entropy, dtype=np.float64
    )

    state.recorded_steps = []
    state.observation_stack = collections.deque()

    # ground banning goes here
    if parameters.wfc_ns.ground != 0 and False:
        pass

    output.previous_decisions = []
    state.previous_decisions = []
    WFC_LOGGER.debug("clear complete")

    return state, output


# A useful helper function which we use because we want numpy arrays instead of jagged arrays
# https://stackoverflow.com/questions/7632963/numpy-find-first-index-of-value-fast/7654768
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1


def find_minimum_entropy(wfc_state, random_variation):
    return None


def find_upper_left_entropy(wfc_state, random_variation):
    return None


def find_upper_left_unresolved(wfc_state, random_variation):
    return None


def find_random_unresolved(wfc_state, random_variation):
    noise_level = 1e-6
    entropy_map = random_variation * noise_level
    entropy_map = entropy_map.flatten() + wfc_state.entropies.flatten()
    minimum_cell = np.argmin(entropy_map)
    if np.any(np.count_nonzero(wfc_state.wave_table, axis=2) == 0):
        WFC_LOGGER.warning("Solver FAIL")
        WFC_LOGGER.debug(f"previous decisions: {len(wfc_state.previous_decisions)}")
        return WFC_FAILURE
    if np.count_nonzero(wfc_state.wave_table, axis=2).flatten()[minimum_cell] == 0:
        WFC_LOGGER.debug(f"previous decisions: {wfc_state}")
        return WFC_FAILURE
    return None

    higher_than_threshold = np.ma.MaskedArray(
        entropy_map, np.count_nonzero(wfc_state.wave_table, axis=2).flatten() <= 1
    )
    minimum_cell = higher_than_threshold.argmin(fill_value=999999.9)
    maximum_cell = higher_than_threshold.argmax(fill_value=0.0)
    chosen_cell = maximum_cell

    if np.any(np.count_nonzero(wfc_state.wave_table, axis=2) == 0):
        WFC_LOGGER.debug("A zero-state node has been found.")

    if wfc_parameters.overflow_check:
        if np.any(np.count_nonzero(wfc_state.wave_table, axis=2) > 65534):
            WFC_LOGGER.error("Overflow A")
            WFC_LOGGER.error(np.count_nonzero(wfc_state.wave_table, axis=2))
            pdb.set_trace()
            assert False
    if np.all(np.count_nonzero(wfc_state.wave_table, axis=2) == 1):
        WFC_LOGGER.info("DETECTED FINISH")
        WFC_LOGGER.info(
            f"nonzero count: {np.count_nonzero(wfc_state.wave_table, axis=2)}"
        )

    cell_index = np.unravel_index(
        chosen_cell,
        [
            np.count_nonzero(wfc_state.wave_table, axis=2).shape[0],
            np.count_nonzero(wfc_state.wave_table, axis=2).shape[1],
        ],
    )
    return CoordRC(row=cell_index[0], column=cell_index[1])


def check_completion(wfc_parameters, wfc_state):
    """Check to see if the solver has failed, found a solution, or should keep going."""
    if np.all(np.count_nonzero(wfc_state.wave_table, axis=2) == 1):
        # Require that every pattern be use at least once?
        pattern_set = set(np.argmax(wfc_state.wave_table, axis=2).flatten())
        # Force a test to encourage backtracking - temporary addition
        if (
            len(pattern_set) != wfc_state.number_of_patterns
        ) and wfc_parameters.wfc_ns.force_use_all_patterns:
            WFC_LOGGER.info("Some patterns were not used")
            return WFC_FAILURE
        WFC_LOGGER.info("Check complete: Solver FINISHED")
        return WFC_FINISHED
    if np.any(np.count_nonzero(wfc_state.wave_table, axis=2) < 1):
        return WFC_FAILURE
    return None


def finalized_observed_waves(parameters, state, output):
    """The solver is finished, process the solution for consumption as part of the output."""
    for row in range(parameters.rows):
        for column in range(parameters.columns):
            pattern_flags = state.wave_table[row, column]
            output.output_grid[row, column] = find_first(
                True, pattern_flags
            )  # TODO: this line is probably overkill?
    state.result = WFC_FINISHED
    return state, output


def make_observation(state, cell, random_number_generator, output):
    return state, output


def wfc_observe(state, random_variation, random_number_generator, parameters, output):
    output.current_iteration_count_observation += 1
    the_result = None
    if np.all(np.count_nonzero(state.wave_table, axis=2) == 1):
        WFC_LOGGER.info("FINISHED")
        WFC_LOGGER.debug(state.wave_table)
        the_result = WFC_FINISHED

    if the_result is None:
        the_result = check_completion(parameters, state)

    cell = None
    if the_result is None:
        if parameters.heuristic == 0:
            cell = find_minimum_entropy(state, random_variation)
        if parameters.heuristic == 1:
            cell = find_upper_left_entropy(state, random_variation)
        if parameters.heuristic == 2:
            cell = find_upper_left_unresolved(state, random_variation)
        if parameters.heuristic == 3:
            cell = find_random_unresolved(state, random_variation)

    if cell is WFC_FAILURE:
        the_result = cell
    if np.all(np.count_nonzero(state.wave_table, axis=2) == 1):
        WFC_LOGGER.info("Solver FINISHED")
        WFC_LOGGER.debug(state.wave_table)
        the_result = WFC_FINISHED
    if the_result is None:
        the_result = check_completion(parameters, state)
    if WFC_FAKE_FAILURE is the_result:
        state.result = WFC_FAILURE
        state.fake_failure = True
        return state, output
    if WFC_FAILURE == the_result:
        state.result = WFC_FAILURE
        return state, output
    if WFC_FINISHED == the_result:
        return finalized_observed_waves(parameters, state, output)

    return make_observation(state, cell, random_number_generator, output)


def is_cell_on_boundary(wfc_parameters, wfc_coords):
    if not wfc_parameters.wfc_ns.periodic_output:
        return False
    # otherwise...
    return False  # TODO


def wrap_coords(wfc_parameters, cell_coords):
    r = (cell_coords.row + wfc_parameters.wfc_ns.generated_size[0]) % (
        wfc_parameters.wfc_ns.generated_size[0]
    )
    c = (cell_coords.column + wfc_parameters.wfc_ns.generated_size[1]) % (
        wfc_parameters.wfc_ns.generated_size[1]
    )
    return CoordRC(row=r, column=c)


def wfc_propagate(parameters, state, output):
    return state, output


def wfc_backtrack(state, output_stack):
    return state, output_stack


BACKTRACK_TRACK_GLOBAL = 0


def reset_backtracking_count():
    global BACKTRACK_TRACK_GLOBAL
    BACKTRACK_TRACK_GLOBAL = 0


def wfc_run(wfc_seed_state, visualize=False, logging=False):
    WFC_LOGGER.info("wfc_run()")
    wfc_output_stack = []
    backtracking_stack = []
    backtracking_count = 0

    wfc_output = types.SimpleNamespace()

    wfc_parameters, wfc_state, wfc_output = wfc_init(wfc_seed_state)
    wfc_parameters.visualize = (visualize,)
    wfc_parameters.logging = logging
    wfc_parameters.timeout = 3000

    wfc_state, wfc_output = wfc_clear(wfc_parameters, wfc_state, wfc_output)
    # if wfc_status.visualize:
    #    show_pattern_adjacency(wfc_state)
    #    visualize_propagator_matrix(wfc_state.propagator_matrix)
    random_number_generator = np.random.RandomState(wfc_parameters.wfc_ns.seed)
    random_variation = random_number_generator.random_sample(wfc_state.entropies.size)
    # record_visualization()
    iterations = 0

    while (iterations < wfc_parameters.wfc_ns.iteration_limit) or (
        0 == wfc_parameters.wfc_ns.iteration_limit
    ):
        wfc_state.backtracking_count = backtracking_count
        wfc_state.backtracking_stack_length = len(backtracking_stack)
        wfc_state.backtracking_total = BACKTRACK_TRACK_GLOBAL
        # if parameters.visualize:
        #    recorded_vis = record_visualization(wfc_state, recorded_vis)
        # wfc_state.current_iteration_count_last_touch += 1
        wfc_state, wfc_output = wfc_observe(
            wfc_state,
            random_variation,
            random_number_generator,
            wfc_parameters,
            wfc_output,
        )

        # Add a time-out on the number of total observations
        print(wfc_output.stats_tracking)
        if wfc_output.stats_tracking["total_observations"] > wfc_parameters.timeout:
            wfc_state.result = WFC_TIMEDOUT
            return wfc_state

        if iterations % 50 == 0:
            print(iterations, end=" ")

        if wfc_parameters.logging:
            with open(wfc_parameters.wfc_ns.debug_log_filename, "a") as stats_file:
                stats_file.write(f"\n=====\n")
                stats_file.write(f"result: {wfc_state.result}\n")
                # stats_file.write(f"total observations: {wfc_state.wfc_ns.stats_tracking['total_observations']}\n")
                stats_file.write(
                    f"On backtracking {backtracking_count}, with stack size {len(backtracking_stack)}\n"
                )
                stats_file.write(f"{wfc_state.result}\n")
                stats_file.write("remaining wave table choices:\n")
                stats_file.write(
                    f"{(np.count_nonzero(wfc_state.wave_table, axis=2))}\n"
                )

        if WFC_FINISHED == wfc_state.result:
            wfc_output.stats_tracking["success"] = True
            # wfc_output.recorded_vis = recorded_vis
            return wfc_state, wfc_output
        if WFC_FAILURE == wfc_state.result:
            if not wfc_parameters.wfc_ns.backtracking:
                return wfc_state, wfc_output
            backtracking_count += 1
            WFC_LOGGER.warning(
                f"Backtracking {backtracking_count}, stack size {len(backtracking_stack)}"
            )
            if backtracking_count > wfc_parameters.wfc_ns.backtracking_limit:
                if wfc_parameters.wfc_ns.backtracking_limit > 0:
                    wfc_state.result = WFC_TIMEDOUT
                    return wfc_state, wfc_output
            wfc_state, backtracking_stack = wfc_backtrack(wfc_state, backtracking_stack)
        wfc_state, wfc_output = wfc_propagate(wfc_parameters, wfc_state, wfc_output)
        iterations += 1
        print(dir(wfc_parameters))
        print("===")
        print(dir(wfc_state))
        assert False
    wfc_state.result = WFC_TIMEDOUT
    print(wfc_state)
    return wfc_state, wfc_output
