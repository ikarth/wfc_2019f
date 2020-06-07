# In[ ]:


wfc_ns_chess48 = types.SimpleNamespace(
    img_filename="Chess.png",  # name of the input file
    seed=87386,  # seed for random generation, can be any number
    tile_size=1,  # size of tile, in pixels
    pattern_width=2,  # Size of the patterns we want. 2x2 is the minimum, larger scales get slower fast.
    channels=3,  # Color channels in the image (usually 3 for RGB)
    adjacency_directions=dict(
        enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)])
    ),  # The list of adjacencies that we care about - these will be turned into the edges of the graph
    periodic_input=True,  # Does the input wrap?
    periodic_output=True,  # Do we want the output to wrap?
    generated_size=(48, 48),  # Size of the final image
    screenshots=1,  # Number of times to run the algorithm, will produce this many distinct outputs
    iteration_limit=0,  # After this many iterations, time out. 0 = never time out.
    allowed_attempts=2,
)  # Give up after this many contradictions
wfc_ns_chess48 = prepare_wfc_namespace(wfc_ns_chess48, visualize=True)
wfc_main(wfc_ns_chess48)


# In[ ]:


wfc_ns_chess47 = types.SimpleNamespace(
    img_filename="Chess.png",  # name of the input file
    seed=87386,  # seed for random generation, can be any number
    tile_size=1,  # size of tile, in pixels
    pattern_width=2,  # Size of the patterns we want. 2x2 is the minimum, larger scales get slower fast.
    channels=3,  # Color channels in the image (usually 3 for RGB)
    adjacency_directions=dict(
        enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)])
    ),  # The list of adjacencies that we care about - these will be turned into the edges of the graph
    periodic_input=True,  # Does the input wrap?
    periodic_output=True,  # Do we want the output to wrap?
    generated_size=(47, 47),  # Size of the final image
    screenshots=1,  # Number of times to run the algorithm, will produce this many distinct outputs
    iteration_limit=0,  # After this many iterations, time out. 0 = never time out.
    allowed_attempts=2,
)  # Give up after this many contradictions
wfc_ns_chess47 = prepare_wfc_namespace(wfc_ns_chess47, visualize=True)
wfc_main(wfc_ns_chess47)


# In[ ]:


wfc_ns_blackdots = types.SimpleNamespace(
    img_filename="blackdots.png",  # name of the input file
    seed=87386,  # seed for random generation, can be any number
    tile_size=1,  # size of tile, in pixels
    pattern_width=2,  # Size of the patterns we want. 2x2 is the minimum, larger scales get slower fast.
    channels=3,  # Color channels in the image (usually 3 for RGB)
    adjacency_directions=dict(
        enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)])
    ),  # The list of adjacencies that we care about - these will be turned into the edges of the graph
    periodic_input=True,  # Does the input wrap?
    periodic_output=True,  # Do we want the output to wrap?
    generated_size=(48, 48),  # Size of the final image
    screenshots=1,  # Number of times to run the algorithm, will produce this many distinct outputs
    iteration_limit=0,  # After this many iterations, time out. 0 = never time out.
    allowed_attempts=2,
)  # Give up after this many contradictions
wfc_ns_blackdots = prepare_wfc_namespace(wfc_ns_blackdots, visualize=True)
wfc_main(wfc_ns_blackdots)


# In[ ]:


wfc_ns_blackdotsred = types.SimpleNamespace(
    img_filename="blackdotsred.png",  # name of the input file
    seed=87386,  # seed for random generation, can be any number
    tile_size=1,  # size of tile, in pixels
    pattern_width=2,  # Size of the patterns we want. 2x2 is the minimum, larger scales get slower fast.
    channels=3,  # Color channels in the image (usually 3 for RGB)
    adjacency_directions=dict(
        enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)])
    ),  # The list of adjacencies that we care about - these will be turned into the edges of the graph
    periodic_input=True,  # Does the input wrap?
    periodic_output=True,  # Do we want the output to wrap?
    generated_size=(48, 48),  # Size of the final image
    screenshots=1,  # Number of times to run the algorithm, will produce this many distinct outputs
    iteration_limit=0,  # After this many iterations, time out. 0 = never time out.
    allowed_attempts=2,
)  # Give up after this many contradictions
wfc_ns_blackdotsred = prepare_wfc_namespace(wfc_ns_blackdotsred, visualize=True)
wfc_main(wfc_ns_blackdotsred)


# In[ ]:


wfc_ns_blackdotsstripe = types.SimpleNamespace(
    img_filename="blackdotsstripe.png",  # name of the input file
    seed=87386,  # seed for random generation, can be any number
    tile_size=1,  # size of tile, in pixels
    pattern_width=2,  # Size of the patterns we want. 2x2 is the minimum, larger scales get slower fast.
    channels=3,  # Color channels in the image (usually 3 for RGB)
    adjacency_directions=dict(
        enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)])
    ),  # The list of adjacencies that we care about - these will be turned into the edges of the graph
    periodic_input=True,  # Does the input wrap?
    periodic_output=True,  # Do we want the output to wrap?
    generated_size=(48, 48),  # Size of the final image
    screenshots=1,  # Number of times to run the algorithm, will produce this many distinct outputs
    iteration_limit=0,  # After this many iterations, time out. 0 = never time out.
    allowed_attempts=2,
)  # Give up after this many contradictions
wfc_ns_blackdotsstripe = prepare_wfc_namespace(wfc_ns_blackdotsstripe, visualize=True)
wfc_main(wfc_ns_blackdotsstripe)


# In[ ]:


wfc_ns_Skyline = types.SimpleNamespace(
    img_filename="Skyline.png",  # name of the input file
    seed=87386,  # seed for random generation, can be any number
    tile_size=1,  # size of tile, in pixels
    pattern_width=2,  # Size of the patterns we want. 2x2 is the minimum, larger scales get slower fast.
    channels=3,  # Color channels in the image (usually 3 for RGB)
    adjacency_directions=dict(
        enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)])
    ),  # The list of adjacencies that we care about - these will be turned into the edges of the graph
    periodic_input=True,  # Does the input wrap?
    periodic_output=True,  # Do we want the output to wrap?
    generated_size=(48, 48),  # Size of the final image
    screenshots=1,  # Number of times to run the algorithm, will produce this many distinct outputs
    iteration_limit=0,  # After this many iterations, time out. 0 = never time out.
    allowed_attempts=2,
)  # Give up after this many contradictions
wfc_ns_Skyline = prepare_wfc_namespace(wfc_ns_Skyline, visualize=True)
wfc_main(wfc_ns_Skyline)


# In[ ]:


wfc_ns_flowers = types.SimpleNamespace(
    img_filename="Flowers.png",  # name of the input file
    seed=87386,  # seed for random generation, can be any number
    tile_size=1,  # size of tile, in pixels
    pattern_width=3,  # Size of the patterns we want. 2x2 is the minimum, larger scales get slower fast.
    channels=3,  # Color channels in the image (usually 3 for RGB)
    adjacency_directions=dict(
        enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)])
    ),  # The list of adjacencies that we care about - these will be turned into the edges of the graph
    periodic_input=True,  # Does the input wrap?
    periodic_output=True,  # Do we want the output to wrap?
    generated_size=(48, 48),  # Size of the final image
    screenshots=1,  # Number of times to run the algorithm, will produce this many distinct outputs
    iteration_limit=0,  # After this many iterations, time out. 0 = never time out.
    allowed_attempts=2,
)  # Give up after this many contradictions
wfc_ns_flowers = prepare_wfc_namespace(wfc_ns_flowers, visualize=True)
wfc_main(wfc_ns_flowers)
