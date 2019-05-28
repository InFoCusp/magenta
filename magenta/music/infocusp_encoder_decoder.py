# Copyright 2018 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classes for converting between event sequences and models inputs/outputs that are implemented by Infocusp.
Base classes taken from magenta.music.encoder_decoder. More description of base classes available in that file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numbers

from magenta.common import sequence_example_lib
from magenta.music import constants
from magenta.pipelines import pipeline
from magenta.music import encoder_decoder
import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
import os

DEFAULT_STEPS_PER_BAR = constants.DEFAULT_STEPS_PER_BAR
DEFAULT_LOOKBACK_DISTANCES = [DEFAULT_STEPS_PER_BAR, DEFAULT_STEPS_PER_BAR * 2]

# Twisha - Added this encoder class which uses the context vector obtained directly from data instead of one hot vector
# Equivalent to OneHotEventSequenceEncoderDecoder
class NextNoteDistributionEventSequenceEncoderDecoder(encoder_decoder.EventSequenceEncoderDecoder):
  """An implementation of EventSequenceEncoderDecoder that produces a one-hot encoding of current note + probability distribution for next note."""

  def __init__(self, one_hot_encoding):
    """Initialize a OneHotEventSequenceEncoderDecoder object.
    Args:
      one_hot_encoding: A OneHotEncoding object that transforms events to and
          from integer indices.
    """
    self._one_hot_encoding = one_hot_encoding

  @property
  def input_size(self):
    return self._one_hot_encoding.num_classes

  @property
  def num_classes(self):
    return self._one_hot_encoding.num_classes

  @property
  def default_event_label(self):
    return self._one_hot_encoding.encode_event(
        self._one_hot_encoding.default_event)

  def events_to_input(self, events, position):
    """Returns the input vector for the given position in the event sequence.
    Returns a one-hot vector for the given position in the event sequence, as
    determined by the one hot encoding which is then augmented with probability distribution of next note.
    This distribution is pre-calculated and stored in a file and loaded here.
    Args:
      events: A list-like sequence of events.
      position: An integer event position in the event sequence.
    Returns:
      An input vector, a list of floats.
    """
    data_dir = os.environ['BASE_DATA_PATH']
    filename = os.path.join(data_dir, 'model_weights/overall_proximity_count.npy')
    overall_proximity_count = np.load(filename)
    # input_ = [0.0] * self.input_size
    index = self._one_hot_encoding.encode_event(events[position])

    input_ = overall_proximity_count[index].copy()
    input_ = np.array(input_, dtype='float')
    input_[0] = input_[0] / input_.sum()
    input_[1:] = input_[1:]/input_[1:].sum()
    input_[index] += 1

    return input_

  def events_to_label(self, events, position):
    """Returns the label for the given position in the event sequence.
    Returns the zero-based index value for the given position in the event
    sequence, as determined by the one hot encoding.
    Args:
      events: A list-like sequence of events.
      position: An integer event position in the event sequence.
    Returns:
      A label, an integer.
    """
    return self._one_hot_encoding.encode_event(events[position])

  def class_index_to_event(self, class_index, events):
    """Returns the event for the given class index.
    This is the reverse process of the self.events_to_label method.
    Args:
      class_index: An integer in the range [0, self.num_classes).
      events: A list-like sequence of events. This object is not used in this
          implementation.
    Returns:
      An event value.
    """
    return self._one_hot_encoding.decode_event(class_index)

  def labels_to_num_steps(self, labels):
    """Returns the total number of time steps for a sequence of class labels.
    Args:
      labels: A list-like sequence of integers in the range
          [0, self.num_classes).
    Returns:
      The total number of time steps for the label sequence, as determined by
      the one-hot encoding.
    """
    events = []
    for label in labels:
      events.append(self.class_index_to_event(label, events))
    return sum(self._one_hot_encoding.event_to_num_steps(event)
               for event in events)
  