from wfc.wfc_utilities import CoordXY, CoordRC, hash_downto, find_pattern_center
from wfc.wfc_tiles import tiles_to_images
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplot, subplots, title, matshow
import numpy as np
import itertools
import math
import matplotlib

# In[15]:


# def is_valid_overlap_xy(d, p1, p2, pattern_catalog, pattern_width, adjacency_directions):
#     """Given a direction, two patterns, and a pattern catalog, return True
#     if we overlap pattern two on top of pattern one and the intersection 
#     of the two patterns is an exact match."""
#     dimensions = (1,0)
#     not_a_number = -1
#     adjacency_directions_inverted = CoordXY(x = 0 - adjacency_directions[d].x, y = 0 - adjacency_directions[d].y)
    
#     ##TODO: can probably speed this up by using the right slices, rather than rolling the whole pattern...
#     shifted = np.roll(np.pad(pattern_catalog[p2], pattern_width, mode='constant', constant_values = not_a_number), adjacency_directions[d], dimensions)
#     #print("*")
#     #print(shifted)
#     compare = shifted[pattern_width:pattern_width+pattern_width, pattern_width:pattern_width+pattern_width]
#     left = max(0,0 + adjacency_directions_inverted.x)
#     right = min(pattern_width, pattern_width + adjacency_directions_inverted.x)
#     top = max(0,0 + adjacency_directions_inverted.y)
#     bottom = min(pattern_width, pattern_width + adjacency_directions_inverted.y)
#     a = pattern_catalog[p1][top:bottom,left:right]
#     b = compare[top:bottom,left:right]
    
#     #a = pattern_catalog[p1]
#     #b = pattern_catalog[p2]
#     #b_shift = np.roll(b, (adjacency_directions[d].y,adjacency_directions[d].x), (0,1))
#     #a_slice =       a[0+adjacency_directions[d].y:pattern_width+1+adjacency_directions[d].y, 0+adjacency_directions[d].x:pattern_width+1+adjacency_directions[d].x] 
#     #b_slice = b_shift[0+adjacency_directions[d].y:pattern_width+1+adjacency_directions[d].y, 0+adjacency_directions[d].x:pattern_width+1+adjacency_directions[d].x] 

#     print(a)
#     print(b)
#     print(compare)
#     print(shifted)
#     res = np.array_equal(a,b)
    
#     print(f"is_valid_overlap: {p1}+{p2} at {d}{adjacency_directions[d]} = {res}")
#     #print(f"\n{a}\n = \n{b}\n{b_shift}")
#     #print(a_slice)
#     #print('<-')
#     #print(b_slice)
    
#     #shifted = np.roll(np.pad(pattern_catalog[p2], 
#     #                         pattern_width, 
#     #                         mode='constant', 
#     #                         constant_values = not_a_number), 
#     #                  adjacency_directions[d], dimensions)
#     #print(a,b)
    
#     return res

def is_valid_overlap_xy(dir_id, p1, p2, pattern_catalog, pattern_width, adjacency_directions):
    """Given a direction, two patterns, and a pattern catalog, return True
    if we overlap pattern two on top of pattern one and the intersection 
    of the two patterns is an exact match."""
    #dir_corrected = (0 - adjacency_directions[dir_id].x, 0 - adjacency_directions[dir_id].y)
    dir_corrected = (0 + adjacency_directions[dir_id].x, 0 + adjacency_directions[dir_id].y)
    dimensions = (1,0)
    not_a_number = -1
    #TODO: can probably speed this up by using the right slices, rather than rolling the whole pattern...
    #print(d, p2, p1)
    shifted = np.roll(np.pad(pattern_catalog[p2], pattern_width, mode='constant', constant_values = not_a_number), dir_corrected, dimensions)
    compare = shifted[pattern_width:pattern_width+pattern_width, pattern_width:pattern_width+pattern_width]
    left = max(0,0 + dir_corrected[0])
    right = min(pattern_width, pattern_width + dir_corrected[0])
    top = max(0,0 + dir_corrected[1])
    bottom = min(pattern_width, pattern_width + dir_corrected[1])
    a = pattern_catalog[p1][top:bottom,left:right]
    b = compare[top:bottom,left:right]
    res = np.array_equal(a,b)
    #print(f"res: {res}")
    return res

def valid_overlap(d, p1, p2):
        dimensions = (1,0)
        not_a_number = 0
        #TODO: can probably speed this up by using the right slices, rather than rolling the whole pattern...
        shifted = numpy.roll(numpy.pad(pattern_catalog[p2], max(patternsize), mode='constant', constant_values = not_a_number), d, dimensions)
        compare = shifted[patternsize[0]:patternsize[0]+patternsize[0], patternsize[1]:patternsize[1]+patternsize[1]]
        left = max(0,0 + d[0])
        right = min(patternsize[0], patternsize[0]+d[0])
        top = max(0,0 + d[1])
        bottom = min(patternsize[1], patternsize[1]+d[1])
        
        a = pattern_catalog[p1][top:bottom,left:right]
        b = compare[top:bottom,left:right]
        res = numpy.array_equal(a,b)
        return res

def adjacency_extraction_consistent(wfc_ns, pattern_cat):
    """Takes a pattern catalog, returns a list of all legal adjacencies."""
    # This is a brute force implementation. We should really use the adjacency list we've already calculated...
    legal = []
    #print(f"pattern_cat\n{pattern_cat}")
    for p1, pattern1 in enumerate(pattern_cat):
        for d_index, d in enumerate(wfc_ns.adjacency_directions):
            for p2, pattern2 in enumerate(pattern_cat):
                if is_valid_overlap_xy(d, p1, p2, pattern_cat, wfc_ns.pattern_width, wfc_ns.adjacency_directions):
                    legal.append((d_index, p1, p2))
    return legal


# ..  ..  *.  .*  ..
# .*  *.  ..  ..  ..

# 1+0
# ...
# .*.

# 1+1
# ...
# *X.

# 1+2
# .X.
# .X.

# 1+3
# .X.
# .X.

# 1+4
# ...
# .X.






# In[16]:




#def adjacency_efficent_extraction_observed(codes):
#    adjacency_relations = list()
#
#    #if mode == 'observed': # just pairing seen in the input image
#    for i,adj_dir in ns.adjacency_directions.items():
#        #print(codes)
#        u = adj_dir.x
#        v = adj_dir.y
#        a = codes[max(0,0+u):codes.shape[0]+u,max(0,0+v):codes.shape[1]+v]
#        b = codes[max(0,0-u):codes.shape[0]-u,max(0,0-v):codes.shape[1]-v]
#        triples = [(i,ns.tile_ids[j],ns.tile_ids[k]) for j,k in zip(a.ravel(),b.ravel())]
#        adjacency_relations.extend(set(triples))
#    return adjacency_relations

import collections
import itertools

def adjacency_efficent_extraction_consistent(patterns):
    assert ns.pattern_width > 0
    adjacency_relations = list()
    for i,adj_dir in ns.adjacency_directions.items():
        a = hash_downto(patterns[:,max(0,0+adj_dir.x):ns.pattern_width+adj_dir.x,max(0,0+adj_dir.y):ns.pattern_width+adj_dir.y],1)
        b = hash_downto(patterns[:,max(0,0-adj_dir.x):ns.pattern_width-adj_dir.x,max(0,0-adj_dir.y):ns.pattern_width-adj_dir.y],1)
        rel = collections.defaultdict(lambda: ([],[]))
        for ia,key in enumerate(a):
            rel[key][0].append(ia)
        for ib,key in enumerate(b):
            rel[key][1].append(ib)
        
        triples = []
        for (ias,ibs) in rel.values():
            adjacency_relations.extend([(i,ia,ib) for ia,ib in itertools.product(ias,ibs)])
    
    return adjacency_relations
#
#"""%%time 
#adjacency_relations2 = adjacency_efficent_extraction_observed(ns.patterns)
#adjacency_relations3 = adjacency_efficent_extraction_consistent(ns.patterns)
#print(adjacency_relations2)
#"""


# In[17]:


#test = np.array([[0,1,2,3],[4,5,6,7]])
#print(test)
#print()
#print(np.repeat(test[:,:,np.newaxis], 4, axis=2))


# In[18]:


def blit(destination, sprite, upper_left, layer = False, check=False):
    """
    Blits one multidimensional array into another numpy array.
    """
    lower_right = [((a + b) if ((a + b) < c) else c) for a,b,c in zip(upper_left, sprite.shape, destination.shape)]
    if min(lower_right) < 0:
        return
    
    for i_index, i in enumerate(range(upper_left[0], lower_right[0])):
        for j_index, j in enumerate(range(upper_left[1], lower_right[1])):
            if (i >= 0) and (j >= 0):
                if len(destination.shape) > 2:
                    destination[i, j, layer] = sprite[i_index, j_index]
                else:
                    if check:
                        if (destination[i, j] == sprite[i_index, j_index]) or (destination[i, j] == -1) or {sprite[i_index, j_index] == -1}:
                            destination[i, j] = sprite[i_index, j_index]
                        else:
                            print("ERROR, mismatch: destination[{i},{j}] = {destination[i, j]}, sprite[{i_index}, {j_index}] = {sprite[i_index, j_index]}")
                    else:
                        destination[i, j] = sprite[i_index, j_index]
    return destination


# In[19]:


import pprint

def show_adjacencies(wfc_ns, adjacency_relations_list):
    try:
        figadj = figure(figsize=(12,1+len(adjacency_relations_list)), edgecolor='b')
        title('Adjacencies')
        max_offset = max([abs(x) for x in list(itertools.chain.from_iterable(wfc_ns.adjacency_directions.values()))])

        for i,adj_rel in enumerate(adjacency_relations_list):
            preview_size = (wfc_ns.pattern_width + max_offset*2)
            preview_adj = np.full((preview_size,preview_size), -1, dtype=np.int64)    
            upper_left_of_center = CoordXY(x=max_offset,y=max_offset)#(ns.pattern_width, ns.pattern_width)
            #print(f"adj_rel: {adj_rel}")
            blit(preview_adj, wfc_ns.patterns[adj_rel[1]], upper_left_of_center, check=True)
            blit(preview_adj, wfc_ns.patterns[adj_rel[2]], 
                 (upper_left_of_center.y + wfc_ns.adjacency_directions[adj_rel[0]].y, 
                  upper_left_of_center.x + wfc_ns.adjacency_directions[adj_rel[0]].x), check=True)

            ptr = tiles_to_images(wfc_ns, preview_adj, wfc_ns.tile_catalog, wfc_ns.tile_size, visualize=True).astype(np.uint8)
            
            subp = subplot(math.ceil(len(adjacency_relations_list) / 4),4, i+1)
            spi = subp.imshow(ptr)
            spi.axes.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            title(f'{i}: ({adj_rel[1]} + {adj_rel[2]}) by\n{wfc_ns.adjacency_directions[adj_rel[0]]}', fontsize=10)
            
            indicator_rect = matplotlib.patches.Rectangle((upper_left_of_center.y - 0.51, upper_left_of_center.x - 0.51), wfc_ns.pattern_width, wfc_ns.pattern_width, Fill=False, edgecolor='b', linewidth=3.0, linestyle=':')
            
            spi.axes.add_artist(indicator_rect)
            spi.axes.grid(False)
        plt.savefig(wfc_ns.output_filename + "_adjacency.pdf", bbox_inches="tight")
        plt.close()
    except ValueError as e:
        print(e)

# # Solvers

# ## Adjacency Grid

# In[20]:


def make_adjacency_grid(shape: tuple, directions):
    adj_grid_shape = ((len(directions)), *shape)
    def within_bounds(x, limit):
        while x < 0:
            x += limit
        while x > limit - 1:
            x -= limit
        return x
    def add_offset(a,b):
        offset = [sum(x) for x in zip(a,b)]
        return (within_bounds(offset[0], shape[0]),within_bounds(offset[1], shape[1]))
    adj_grid = np.zeros(adj_grid_shape, dtype = np.uint32) 
    
    # TODO: grids bigger than max(uint32 - 1) can't index the entire array
    # Therefore, output shapes should not exceed approximately 65535 x 65535
    for d_idx, d in directions.items():
        for y in range(shape[0]):
            for x in range(shape[1]):
                cell = (y, x)
                offset_cell = add_offset(d, cell)
                adj_grid[d_idx,y,x] = (offset_cell[0] + (offset_cell[1] * shape[0]))
                # Adds one to the index so we can use zero in MiniZinc to 
                # indicate non-edges for non-wrapping output images and similar
                #adj_grid[d_idx,x,y] = 1 + (offset_cell[0] + (offset_cell[1] * shape[0]))
    return adj_grid

def make_reverse_adjacency_directions(adjacency_directions):
    reverse_adjacency_directions_val = {}
    for k,v in adjacency_directions.items():
        reverse_adjacency_directions_val[k] = tuple([(i * -1) for i in v])
        #print(f"reverse_adjacency direction {v} to {reverse_adjacency_directions_val[k]}")
    return reverse_adjacency_directions_val#, reverse_directions_by_index

def make_reverse_adjacency_grid(shape, directions):
  reverse_directions = make_reverse_adjacency_directions(directions)
  return make_adjacency_grid(shape, reverse_directions)

def adjacency_index(shape, index_num):
  x = math.floor((index_num) % shape[0])
  y = math.floor((index_num) / shape[0])
  return (x, y)

def reverse_direction_index(directions):
  rev_directions = make_reverse_adjacency_directions(directions)#[tuple([(i * -1) for i in o]) for o in directions.values()]
  inverted_offset = {}
  # Match with first offset value that matches
  for rev_key, rev_val in rev_directions.items():
      for key, val in directions.items():
        if val == rev_val:
          inverted_offset[rev_key] = key
          break
  return inverted_offset

def get_direction_from_offset(rdirections, offset):
  # TODO: Currently assumes a one-to-one mapping, which dictionaries do not enforce.
  for rev_key, rev_val in rdirections.items():
    if rev_val == offset:
      return rev_key
  raise ValueError("Offset not found in directions.")
  


# In[21]:


#adjacency_grid = make_adjacency_grid(wfc_ns_chess.generated_size, wfc_ns_chess.adjacency_directions)
#print(wfc_ns_chess.adjacency_directions)
#reverse_adjacency_directions = make_reverse_adjacency_directions(wfc_ns_chess.adjacency_directions)
#print(reverse_adjacency_directions)
#reverse_adjacency_grid = make_adjacency_grid(wfc_ns_chess.generated_size, reverse_adjacency_directions)
#adjacency_grid = make_adjacency_grid(wfc_ns_chess.generated_size, wfc_ns_chess.adjacency_directions)
#reverse_adjacency_grid = make_reverse_adjacency_grid(wfc_ns_chess.generated_size, wfc_ns_chess.adjacency_directions)

#print(f"reverse: {reverse_direction_index(wfc_ns_chess.adjacency_directions)}")

#print(wfc_ns_chess.adjacency_directions)
##print(adjacency_grid)
##print(reverse_adjacency_grid)
#print(adjacency_index(wfc_ns_chess.generated_size, 2))
#print(reverse_direction_index(wfc_ns_chess.adjacency_directions))
#print(reverse_direction_index(wfc_ns_chess.adjacency_directions)[get_direction_from_offset(wfc_ns_chess.adjacency_directions, (0,-1))])





if __name__ == "__main__":
    import types
    test_ns = types.SimpleNamespace(img_filename = "red_maze.png", seed = 87386, tile_size = 1, pattern_width = 2, channels = 3, adjacency_directions = dict(enumerate([CoordXY(x=0,y=-1),CoordXY(x=1,y=0),CoordXY(x=0,y=1),CoordXY(x=-1,y=0)])), periodic_input = True, periodic_output = True, generated_size = (3,3), screenshots = 1, iteration_limit = 0, allowed_attempts = 1) 
    test_ns = wfc_utilities.find_pattern_center(test_ns)
    test_ns = wfc_utilities.load_visualizer(test_ns)
    test_ns.img = load_source_image(test_ns.img_filename)
    test_ns.tile_catalog, test_ns.tile_grid, test_ns.code_list, test_ns.unique_tiles = make_tile_catalog(test_ns)
    test_ns.tile_ids = {v:k for k,v in dict(enumerate(test_ns.unique_tiles[0])).items()}
    test_ns.tile_weights = {a:b for a,b in zip(test_ns.unique_tiles[0], test_ns.unique_tiles[1])}
    import doctest
    doctest.testmod()




