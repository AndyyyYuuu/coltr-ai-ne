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

BITE_SIZE = 1024 # When modifying bite size, keep time resolution in mind
TIME_RESOLUTION = 32

# Load the MIDI file
def midi_to_tensor(midi_file):
    notes = midi_file.instruments[0].notes # Grab notes only
    time_resolution = TIME_RESOLUTION  # Quantize

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


def pad_tensor(tensor, target_size):
    current_size = tensor.shape[0]
    if current_size == target_size:
        return tensor
    elif current_size > target_size:
        raise ValueError(f"Input tensor exceeds target size {target_size}")
    zeros_num = target_size - current_size
    zeros = torch.zeros((zeros_num,) + tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, zeros), dim=0)


class JazzDataset(Dataset):
    def __init__(self, data_dir, chunk_size):
        print("Initializing dataset...")
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.data = []
        for i in sorted(os.listdir(data_dir)):
            midi_tensor = midi_to_tensor(pretty_midi.PrettyMIDI(os.path.join(self.data_dir, i)))
            self.data.extend([pad_tensor(t, BITE_SIZE) for t in torch.split(midi_tensor, self.chunk_size)])
        print("Dataset complete!")
        '''
        for i in sorted(os.listdir(data_dir)):
            self.data.append(torch.split(midi_to_tensor(pretty_midi.PrettyMIDI(os.path.join(self.data_dir, i))), self.chunk_size))
        '''

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        '''
        path = os.path.join(self.data_dir, self.data[i])
        midi_file = pretty_midi.PrettyMIDI(path)
        print(path)
        '''
        return self.data[i]
        #return midi_to_tensor(self.data[i])

dataset = JazzDataset(data_dir="weimar_jazz_database", chunk_size=BITE_SIZE)
#print(torch.split(item, BITE_SIZE))

train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
batch_size = 32
# Create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

subset_tensor = dataset.__getitem__(-1)#[:1000, :]
print(subset_tensor.shape)

# Create a binary image-like visualization
plt.imshow(numpy.flipud(subset_tensor.T), cmap='gray', aspect='auto')

# Add labels to the axes
plt.xlabel('Time Step')
plt.ylabel('MIDI note #')
plt.title('')

# Show the plot
plt.show()

# ------ END OF DATA PIPELINE CODE --------
