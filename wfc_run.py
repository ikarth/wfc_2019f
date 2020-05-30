# -*- coding: utf-8 -*-
"""Base code to load commands from xml and run them."""

import time
import wfc.wfc_control as wfc_control
import xml.etree.ElementTree as ET
import os

def string2bool(strn):
    if isinstance(strn, bool):
        return strn
    return strn.lower() in ["true"]

def run_default(run_experiment=False):
    log_filename = f"log_{time.time()}"
    xdoc = ET.ElementTree(file="commands.xml")
    default_allowed_attempts = 10
    default_backtracking = False
    log_stats_to_output = wfc_control.make_log_stats()
    visualize_experiment = False

    for xnode in xdoc.getroot():
        name = xnode.get('name', "NAME")
        if "precache" == xnode.tag:
            precache = xnode.get('precache', "ERROR")
            if "ERROR" == precache:
              console.error(f"file path to precache missing: {name}")
            tile_size = int(xnode.get('tile_size', 1)) # size of tile, in pixels
            pattern_width = int(xnode.get('N', 2)) # Size of the patterns we want.
            symmetry = int(xnode.get('symmetry', 8))
            ground = int(xnode.get('ground', 0))
            periodic_input = string2bool(xnode.get('periodic', False)) # Does the input wrap?
            periodic_output = string2bool(xnode.get('periodic', False)) # Do we want the output to wrap?
            generated_size = (int(xnode.get('width', 48)), int(xnode.get('height', 48)))
            screenshots = int(xnode.get('screenshots', 3)) # Number of times to run the algorithm, will produce this many distinct outputs
            iteration_limit = int(xnode.get('iteration_limit', 0)) # After this many iterations, time out. 0 = never time out.
            allowed_attempts = int(xnode.get('allowed_attempts', default_allowed_attempts)) # Give up after this many contradictions
            backtracking = string2bool(xnode.get('backtracking', default_backtracking))
            choice_heuristic=xnode.get('choice_heuristic', "weighted")
            loc_heuristic=xnode.get('loc_heuristic', "entropy")
            global_constraint=xnode.get('global_constraint', "None")

            for x in range(screenshots):
                print(f"-: {name} > {x}")
                solution = wfc_control.execute_wfc(name,
                                                   use_precache=precache,
                                                   tile_size=tile_size,
                                                   pattern_width=pattern_width,
                                                   rotations=symmetry,
                                                   output_size=generated_size,
                                                   ground=ground,
                                                   attempt_limit=allowed_attempts,
                                                   output_periodic=periodic_output,
                                                   input_periodic=periodic_input,
                                                   loc_heuristic=loc_heuristic,
                                                   choice_heuristic=choice_heuristic,
                                                   backtracking=backtracking,
                                                   global_constraint=global_constraint,
                                                   log_filename=log_filename,
                                                   log_stats_to_output=log_stats_to_output,
                                                   visualize=visualize_experiment,
                                                   logging=False,
                                                   save_precache=False,
                                                   execute_solver=True
                )
                if solution is None:
                    print(None)
                else:
                    print(solution)


        if "overlapping" == xnode.tag:
            #seed = 3262 # seed for random generation, can be any number
            tile_size = int(xnode.get('tile_size', 1)) # size of tile, in pixels
            pattern_width = int(xnode.get('N', 2)) # Size of the patterns we want.
            # 2x2 is the minimum, larger scales get slower fast.

            symmetry = int(xnode.get('symmetry', 8))
            ground = int(xnode.get('ground', 0))
            periodic_input = string2bool(xnode.get('periodic', False)) # Does the input wrap?
            periodic_output = string2bool(xnode.get('periodic', False)) # Do we want the output to wrap?
            generated_size = (int(xnode.get('width', 48)), int(xnode.get('height', 48)))
            screenshots = int(xnode.get('screenshots', 3)) # Number of times to run the algorithm, will produce this many distinct outputs
            iteration_limit = int(xnode.get('iteration_limit', 0)) # After this many iterations, time out. 0 = never time out.
            allowed_attempts = int(xnode.get('allowed_attempts', default_allowed_attempts)) # Give up after this many contradictions
            backtracking = string2bool(xnode.get('backtracking', default_backtracking))

            run_instructions = [{"loc": "entropy", "choice": "weighted", "backtracking":backtracking, "global": None}]
            #run_instructions = [{"loc": "entropy", "choice": "weighted", "backtracking": True, "global": "allpatterns"}]
            if run_experiment:
                run_instructions = [{"loc": "lexical", "choice": "weighted", "backtracking":backtracking, "global": None},
                                    {"loc": "entropy", "choice": "weighted", "backtracking":backtracking, "global": None},
                                    {"loc": "random",  "choice": "weighted", "backtracking":False, "global": None},
                                    {"loc": "lexical", "choice": "random",  "backtracking":backtracking, "global": None},
                                    {"loc": "entropy", "choice": "random",  "backtracking":backtracking, "global": None},
                                    {"loc": "random",  "choice": "random",  "backtracking":False, "global": None},
                                    {"loc": "lexical", "choice": "weighted", "backtracking":True, "global": None},
                                    {"loc": "entropy", "choice": "weighted", "backtracking":True, "global": None},
                                    {"loc": "lexical", "choice": "weighted", "backtracking":True, "global": "allpatterns"},
                                    {"loc": "entropy", "choice": "weighted", "backtracking":True, "global": "allpatterns"},
                                    {"loc": "lexical", "choice": "weighted", "backtracking":False, "global": "allpatterns"},
                                    {"loc": "entropy", "choice": "weighted", "backtracking":False, "global": "allpatterns"}]
            if run_experiment == "heuristic":
                run_instructions = [
                    {"loc": "hilbert", "choice": "weighted", "backtracking":backtracking, "global": None},
                    {"loc": "spiral",  "choice": "weighted", "backtracking":backtracking, "global": None},
                    {"loc": "entropy", "choice": "weighted", "backtracking":backtracking, "global": None},
                    {"loc": "anti-entropy", "choice": "weighted", "backtracking":backtracking, "global": None},
                    {"loc": "lexical", "choice": "weighted", "backtracking":backtracking, "global": None},
                    {"loc": "simple",  "choice": "weighted", "backtracking":backtracking, "global": None},
                    {"loc": "random",  "choice": "weighted", "backtracking":backtracking, "global": None}
                ]
            if run_experiment == "backtracking":
                run_instructions = [{"loc": "entropy", "choice": "weighted", "backtracking": True,  "global": "allpatterns"},
                                    {"loc": "entropy", "choice": "weighted", "backtracking": False, "global": "allpatterns"},
                                    {"loc": "entropy", "choice": "weighted", "backtracking": True,  "global": None},
                                    {"loc": "entropy", "choice": "weighted", "backtracking": False, "global": None},]
            if run_experiment == "backtracking_heuristic":
                run_instructions = [{"loc": "lexical", "choice": "weighted", "backtracking": True,  "global": "allpatterns"},
                                    {"loc": "lexical", "choice": "weighted", "backtracking": False, "global": "allpatterns"},
                                    {"loc": "lexical", "choice": "weighted", "backtracking": True,  "global": None},
                                    {"loc": "lexical", "choice": "weighted", "backtracking": False, "global": None},
                                    {"loc": "random", "choice": "weighted", "backtracking": True,  "global": "allpatterns"},
                                    {"loc": "random", "choice": "weighted", "backtracking": False, "global": "allpatterns"},
                                    {"loc": "random", "choice": "weighted", "backtracking": True,  "global": None},
                                    {"loc": "random", "choice": "weighted", "backtracking": False, "global": None},]
            if run_experiment == "choices":
                run_instructions = [{"loc": "entropy", "choice": "rarest", "backtracking": False,  "global": None},
                                    {"loc": "entropy", "choice": "weighted", "backtracking": False,  "global": None},
                                    {"loc": "entropy", "choice": "random", "backtracking": False, "global": None},]

            for experiment in run_instructions:
                for x in range(screenshots):
                    print(f"-: {name} > {x}")
                    solution = wfc_control.execute_wfc(name,
                                                       tile_size=tile_size,
                                                       pattern_width=pattern_width,
                                                       rotations=symmetry,
                                                       output_size=generated_size,
                                                       ground=ground,
                                                       attempt_limit=allowed_attempts,
                                                       output_periodic=periodic_output,
                                                       input_periodic=periodic_input,
                                                       loc_heuristic=experiment["loc"],
                                                       choice_heuristic=experiment["choice"],
                                                       backtracking=experiment["backtracking"],
                                                       global_constraint=experiment["global"],
                                                       log_filename=log_filename,
                                                       log_stats_to_output=log_stats_to_output,
                                                       visualize=visualize_experiment,
                                                       logging=False,
                                                       save_precache=True,
                                                       execute_solver=True
                    )
                    if solution is None:
                        print(None)
                    else:
                        print(solution)

            # These are included for my colab experiments, remove them if you're not me
            # os.system('cp -rf "/content/wfc/output/*.tsv" "/content/drive/My Drive/wfc_exper/2"')
            # os.system('cp -r "/content/wfc/output" "/content/drive/My Drive/wfc_exper/2"')

run_default(False)
