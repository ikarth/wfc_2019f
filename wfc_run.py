# -*- coding: utf-8 -*-

import wfc.wfc_control as wfc_control
import xml.etree.ElementTree as ET

def string2bool(strn):
    if isinstance(strn, bool):
        return strn
    return strn.lower() in ["true"]



xdoc = ET.ElementTree(file="samples_test_ground.xml")
default_allowed_attempts = 10
default_backtracking = False
for xnode in xdoc.getroot():
    name = xnode.get('name', "NAME")
    if "overlapping" == xnode.tag:
        #seed = 3262
        tile_size = int(xnode.get('tile_size', 1))
        # seed for random generation, can be any number
        tile_size = int(xnode.get('tile_size', 1)) # size of tile, in pixels
        pattern_width = int(xnode.get('N', 2)) # Size of the patterns we want.
        # 2x2 is the minimum, larger scales get slower fast.

        symmetry = int(xnode.get('symmetry', 8))
        ground = int(xnode.get('ground',0))
        periodic_input = string2bool(xnode.get('periodic', False)) # Does the input wrap?
        periodic_output = string2bool(xnode.get('periodic', False)) # Do we want the output to wrap?
        generated_size = (int(xnode.get('width', 48)),int(xnode.get('height', 48)))
        screenshots = int(xnode.get('screenshots', 3)) # Number of times to run the algorithm, will produce this many distinct outputs
        iteration_limit = int(xnode.get('iteration_limit', 0)) # After this many iterations, time out. 0 = never time out.
        allowed_attempts = int(xnode.get('allowed_attempts', default_allowed_attempts)) # Give up after this many contradictions
        backtracking = string2bool(xnode.get('backtracking', default_backtracking))
        for x in range(screenshots):
            print(f"-: {name} > {x}")
            solution = wfc_control.execute_wfc(name, tile_size=tile_size, pattern_width=pattern_width, rotations=symmetry, output_size=generated_size, ground=ground, output_periodic=periodic_output, input_periodic=periodic_input)
            if solution is None:
                print(None)
            else:
                print(solution)
                
