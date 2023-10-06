# Andy Yu
# October 2023


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, random_split
from struct import pack, unpack
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import pretty_midi
import os
import numpy
# Load the MIDI file
def midi_to_tensor(midi_file):
    notes = midi_file.instruments[0].notes # Grab notes only
    time_resolution = 32  # Quantize

    # Create tensor for MIDI: shape of (num_time_steps, num_pitches)
    num_time_steps = int(midi_file.get_end_time() * time_resolution) + 1
    num_pitches = 128  # Assuming MIDI standard pitch range
    midi_tensor = torch.zeros(num_time_steps, num_pitches)

    # Go through all notes
    for note in notes:
        start_time = int(note.start * time_resolution)
        end_time = int(note.end * time_resolution)
        pitch = note.pitch
        #velocity = note.velocity
        # Fill in the tensor with note
        midi_tensor[start_time:end_time, pitch] = 1

    midi_tensor /= 1.0 # Scaling
    midi_tensor = torch.Tensor(midi_tensor)
    return midi_tensor

class JazzDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = os.listdir(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = os.path.join(self.data_dir, self.data[i])
        midi_file = pretty_midi.PrettyMIDI(path)
        print(path)
        return midi_to_tensor(midi_file)

dataset = JazzDataset(data_dir="weimar_jazz_database")
print(dataset.__getitem__(2))

train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
batch_size = 32
# Create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

subset_tensor = dataset.__getitem__(0)#[:1000, :]  # Visualize the first 1000 time steps as an example

# Create a binary image-like visualization
plt.imshow(numpy.flipud(subset_tensor.T), cmap='gray', aspect='auto')

# Add labels to the axes
plt.xlabel('Time Step')
plt.ylabel('Pitch (MIDI Note Number)')

# Show the plot
plt.show()

