import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Wave(layers.Layer):
  def __init__(self, units=32):
    super(Wave, self).__init__()
    self.units=units

  def call(self, inputs):
    return tf.matmul()

def run(wave, adj_offsets, adj_matrix, locationHeuristic, patternHeuristic, periodic=False, backtracking=False, onBacktrack=None, onChoice=None, onObserve=None, onPropagate=None, checkFeasible=None, onFinal=None, depth=0, depth_limit=None):
  print(wave.shape)
  print(wave.dtype)
  print(adj_offsets)
  print(adj_matrix.shape)
  print(adj_matrix.dtype)
  padded_wave = tf.pad(wave,((0,0),(1,1),(1,1)), mode='REFLECT')
  shifted = [None] * len(adj_offsets)
  for d_count, d_dir in adj_offsets:
    dx, dy = d_dir
    shifted[d_count] = padded_wave[:,1+dx:1+wave.shape[1]+dx,1+dy:1+wave.shape[2]+dy]
    print(shifted[d_count].shape)
  adjacency_stack = tf.stack(shifted, axis=0)

  inputs = keras.Input(shape=wave.shape, dtype=bool, tensor=tf.convert_to_tensor(wave, dtype=bool), name='wave')
  adj_tensor = keras.Input(shape=adjacency_stack.shape, dtype=bool, tensor=adjacency_stack, name='adjacency')



  import pdb; pdb.set_trace()
  return np.argmax(wave, 0)
