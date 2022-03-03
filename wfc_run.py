# -*- coding: utf-8 -*-
"""Base code to load commands from xml and run them."""
from __future__ import annotations

import argparse
import datetime
import logging
from typing import List, Literal, TypedDict, Union
import wfc.wfc_control as wfc_control
import xml.etree.ElementTree as ET
import os


class RunInstructions(TypedDict):
    loc: Literal["lexical", "hilbert", "spiral", "entropy", "anti-entropy", "simple", "random"]
    choice: Literal["lexical", "rarest", "weighted", "random"]
    backtracking: bool
    global_constraint: Literal[False, "allpatterns"]


def string2bool(strn: Union[bool, str]) -> bool:
    if isinstance(strn, bool):
        return strn
    return strn.lower() in ["true"]


def run_default(run_experiment: str = "simple", samples: str = "samples_reference.xml") -> None:
    log_filename = f"log_{datetime.datetime.now().isoformat()}".replace(":", ".")
    xdoc = ET.ElementTree(file=samples)
    default_allowed_attempts = 10
    default_backtracking = str(False)
    log_stats_to_output = wfc_control.make_log_stats()

    for xnode in xdoc.getroot():
        name = xnode.get("name", "NAME")
        if "overlapping" == xnode.tag:
            # seed = 3262
            tile_size = int(xnode.get("tile_size", 1))
            # seed for random generation, can be any number
            tile_size = int(xnode.get("tile_size", 1))  # size of tile, in pixels
            pattern_width = int(xnode.get("N", 2))  # Size of the patterns we want.
            # 2x2 is the minimum, larger scales get slower fast.

            symmetry = int(xnode.get("symmetry", 8))
            ground = int(xnode.get("ground", 0))
            periodic_input = string2bool(
                xnode.get("periodic", "False")
            )  # Does the input wrap?
            periodic_output = string2bool(
                xnode.get("periodic", "False")
            )  # Do we want the output to wrap?
            generated_size = (int(xnode.get("width", 48)), int(xnode.get("height", 48)))
            screenshots = int(
                xnode.get("screenshots", 1)
            )  # Number of times to run the algorithm, will produce this many distinct outputs
            iteration_limit = int(
                xnode.get("iteration_limit", 0)
            )  # After this many iterations, time out. 0 = never time out.
            allowed_attempts = int(
                xnode.get("allowed_attempts", default_allowed_attempts)
            )  # Give up after this many contradictions
            backtracking = string2bool(xnode.get("backtracking", default_backtracking))
            visualize_experiment = False

            run_instructions: List[RunInstructions] = [  # simple
                {
                    "loc": "entropy",
                    "choice": "weighted",
                    "backtracking": backtracking,
                    "global_constraint": False,
                }
            ]
            # run_instructions = [{"loc": "entropy", "choice": "weighted", "backtracking": True, "global_constraint": "allpatterns"}]
            if run_experiment == "choice":
                run_instructions = [
                    {
                        "loc": "lexical",
                        "choice": "weighted",
                        "backtracking": backtracking,
                        "global_constraint": False,
                    },
                    {
                        "loc": "entropy",
                        "choice": "weighted",
                        "backtracking": backtracking,
                        "global_constraint": False,
                    },
                    {
                        "loc": "random",
                        "choice": "weighted",
                        "backtracking": False,
                        "global_constraint": False,
                    },
                    {
                        "loc": "lexical",
                        "choice": "random",
                        "backtracking": backtracking,
                        "global_constraint": False,
                    },
                    {
                        "loc": "entropy",
                        "choice": "random",
                        "backtracking": backtracking,
                        "global_constraint": False,
                    },
                    {
                        "loc": "random",
                        "choice": "random",
                        "backtracking": False,
                        "global_constraint": False,
                    },
                    {
                        "loc": "lexical",
                        "choice": "weighted",
                        "backtracking": True,
                        "global_constraint": False,
                    },
                    {
                        "loc": "entropy",
                        "choice": "weighted",
                        "backtracking": True,
                        "global_constraint": False,
                    },
                    {
                        "loc": "lexical",
                        "choice": "weighted",
                        "backtracking": True,
                        "global_constraint": "allpatterns",
                    },
                    {
                        "loc": "entropy",
                        "choice": "weighted",
                        "backtracking": True,
                        "global_constraint": "allpatterns",
                    },
                    {
                        "loc": "lexical",
                        "choice": "weighted",
                        "backtracking": False,
                        "global_constraint": "allpatterns",
                    },
                    {
                        "loc": "entropy",
                        "choice": "weighted",
                        "backtracking": False,
                        "global_constraint": "allpatterns",
                    },
                ]
            if run_experiment == "heuristic":
                run_instructions = [
                    {
                        "loc": "hilbert",
                        "choice": "weighted",
                        "backtracking": backtracking,
                        "global_constraint": False,
                    },
                    {
                        "loc": "spiral",
                        "choice": "weighted",
                        "backtracking": backtracking,
                        "global_constraint": False,
                    },
                    {
                        "loc": "entropy",
                        "choice": "weighted",
                        "backtracking": backtracking,
                        "global_constraint": False,
                    },
                    {
                        "loc": "anti-entropy",
                        "choice": "weighted",
                        "backtracking": backtracking,
                        "global_constraint": False,
                    },
                    {
                        "loc": "lexical",
                        "choice": "weighted",
                        "backtracking": backtracking,
                        "global_constraint": False,
                    },
                    {
                        "loc": "simple",
                        "choice": "weighted",
                        "backtracking": backtracking,
                        "global_constraint": False,
                    },
                    {
                        "loc": "random",
                        "choice": "weighted",
                        "backtracking": backtracking,
                        "global_constraint": False,
                    },
                ]
            if run_experiment == "backtracking":
                run_instructions = [
                    {
                        "loc": "entropy",
                        "choice": "weighted",
                        "backtracking": True,
                        "global_constraint": "allpatterns",
                    },
                    {
                        "loc": "entropy",
                        "choice": "weighted",
                        "backtracking": False,
                        "global_constraint": "allpatterns",
                    },
                    {
                        "loc": "entropy",
                        "choice": "weighted",
                        "backtracking": True,
                        "global_constraint": False,
                    },
                    {
                        "loc": "entropy",
                        "choice": "weighted",
                        "backtracking": False,
                        "global_constraint": False,
                    },
                ]
            if run_experiment == "backtracking_heuristic":
                run_instructions = [
                    {
                        "loc": "lexical",
                        "choice": "weighted",
                        "backtracking": True,
                        "global_constraint": "allpatterns",
                    },
                    {
                        "loc": "lexical",
                        "choice": "weighted",
                        "backtracking": False,
                        "global_constraint": "allpatterns",
                    },
                    {
                        "loc": "lexical",
                        "choice": "weighted",
                        "backtracking": True,
                        "global_constraint": False,
                    },
                    {
                        "loc": "lexical",
                        "choice": "weighted",
                        "backtracking": False,
                        "global_constraint": False,
                    },
                    {
                        "loc": "random",
                        "choice": "weighted",
                        "backtracking": True,
                        "global_constraint": "allpatterns",
                    },
                    {
                        "loc": "random",
                        "choice": "weighted",
                        "backtracking": False,
                        "global_constraint": "allpatterns",
                    },
                    {
                        "loc": "random",
                        "choice": "weighted",
                        "backtracking": True,
                        "global_constraint": False,
                    },
                    {
                        "loc": "random",
                        "choice": "weighted",
                        "backtracking": False,
                        "global_constraint": False,
                    },
                ]
            if run_experiment == "choices":
                run_instructions = [
                    {
                        "loc": "entropy",
                        "choice": "rarest",
                        "backtracking": False,
                        "global_constraint": False,
                    },
                    {
                        "loc": "entropy",
                        "choice": "weighted",
                        "backtracking": False,
                        "global_constraint": False,
                    },
                    {
                        "loc": "entropy",
                        "choice": "random",
                        "backtracking": False,
                        "global_constraint": False,
                    },
                ]

            for experiment in run_instructions:
                for x in range(screenshots):
                    print(f"-: {name} > {x}")
                    try:
                        solution = wfc_control.execute_wfc(
                            name,
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
                            global_constraint=experiment["global_constraint"],
                            log_filename=log_filename,
                            log_stats_to_output=log_stats_to_output,
                            visualize=visualize_experiment,
                            logging=True,
                        )
                        print(solution)
                    except Exception as exc:
                        print(f"Skipped because: {exc}")

            if False:  # These are included for my colab experiments, remove them if you're not me
                os.system(
                    'cp -rf "/content/wfc/output/*.tsv" "/content/drive/My Drive/wfc_exper/2"'
                )
                os.system(
                    'cp -r "/content/wfc/output" "/content/drive/My Drive/wfc_exper/2"'
                )

def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        description="Geneates examples from bundled samples which will be saved to the output/ directory.",
    )
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        default="simple",
        choices=["simple", "choice", "choices", "heuristic", "backtracking", "backtracking_heuristic"],
        help="Which experiment to run, defaults to simple.",
    )
    parser.add_argument(
        "-s", "--samples",
        type=str,
        required=True,
        metavar="XML_FILE",
        default="samples_reference.xml",
        help="An XML file with input data.  If unsure then use '-s samples_reference.xml'",
    )
    args = parser.parse_args()
    run_default(run_experiment=args.experiment, samples=args.samples)


if __name__ == "__main__":
    main()
