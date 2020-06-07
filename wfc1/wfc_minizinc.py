# An interface to other solvers via MiniZinc
import pymzn
import time
import wfc.wfc_solver
from wfc.wfc_adjacency import adjacency_extraction_consistent
from wfc.wfc_utilities import CoordXY, CoordRC, hash_downto, find_pattern_center


import logging

logging.basicConfig(level=logging.INFO)
wfc_logger = logging.getLogger()

import pdb


def calculate_adjacency_grid(shape: tuple, directions):
    adj_grid_shape = ((len(directions)), *shape)

    def within_bounds(x, limit):
        while x < 0:
            x += limit
        while x > limit - 1:
            x -= limit
        return x

    def add_offset(a, b):
        offset = [sum(x) for x in zip(a, b)]
        return (within_bounds(offset[0], shape[0]), within_bounds(offset[1], shape[1]))

    adj_grid = numpy.zeros(adj_grid_shape, dtype=numpy.uint32)
    # TODO: grids bigger than max(uint32 - 1) can't index the entire array
    # Therefore, output shapes should not exceed approximately 65535 x 65535
    for d_idx, d in enumerate(directions):
        for x in range(shape[0]):
            for y in range(shape[1]):
                cell = (x, y)
                offset_cell = add_offset(d, cell)
                # Adds one to the index so we can use zero in MiniZinc to
                # indicate non-edges for non-wrapping output images and similar
                adj_grid[d_idx, y, x] = 1 + (
                    offset_cell[0] + (offset_cell[1] * shape[0])
                )
    return adj_grid


solns = pymzn.minizinc("knapsack.mzn", "knapsack.dzn", data={"capacity": 20})
print(solns)

pymzn.dict2dzn(
    {
        "w": 8,
        "h": 8,
        "pattern_count": 95,
        "direction_count": 4,
        "adjacency_count": 992,
        "pattern_names": [],
        "relation_matrix": [],
        "adjaceny_table": [],
        "adjacency_matrix": [],
    },
    fout="test.dzn",
)


def mz_init(prestate):
    prestate.adjacency_directions_rc = {
        i: CoordRC(a.y, a.x) for i, a in prestate.adjacency_directions.items()
    }
    prestate = wfc.wfc_utilities.find_pattern_center(prestate)
    wfc_state = types.SimpleNamespace(wfc_ns=prestate)

    wfc_state.result = None
    wfc_state.adjacency_relations = adjacency_extraction_consistent(
        wfc_state.wfc_ns, wfc_state.wfc_ns.patterns
    )
    wfc_state.patterns = np.array(list(wfc_state.wfc_ns.pattern_catalog.keys()))
    wfc_state.pattern_translations = list(wfc_state.wfc_ns.pattern_catalog.values())
    wfc_state.number_of_patterns = wfc_state.patterns.size
    wfc_state.number_of_directions = len(wfc_state.wfc_ns.adjacency_directions)

    wfc_state.propagator_matrix = np.zeros(
        (
            wfc_state.number_of_directions,
            wfc_state.number_of_patterns,
            wfc_state.number_of_patterns,
        ),
        dtype=np.bool_,
    )
    for d, p1, p2 in wfc_state.adjacency_relations:
        wfc_state.propagator_matrix[(d, p1, p2)] = True

    wfc_state.mz_dzn = {
        "w": wfc_state.wfc_ns.generated_size[0],
        "h": wfc_state.wfc_ns.generated_size[1],
        "pattern_count": wfc_state.number_of_patterns,
        "direction_count": wfc_state.number_of_directions,
        "adjacency_count": len(wfc_state.adjacency_relations),
        "pattern_names": list(
            wfc_state.patterns, zip(range(wfc_state.number_of_patterns))
        ),
        "relation_matrix": calculate_adjacency_grid(
            wfc_state.wfc_ns.generated_size, wfc_state.wfc_ns.adjacency_directions
        ),
        "adjaceny_table": [],
        "adjacency_matrix": [],
    }


def mz_run(wfc_seed_state):
    wfc_logger.info("Invoking MiniZinc solver")
    wfc_state = mz_init(wfc_seed_state)

    return wfc_state
