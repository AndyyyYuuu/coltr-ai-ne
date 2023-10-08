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
NUM_PITCHES = 128
HIDDEN_SIZE = 512

# ----- End of Imports -----
# ----- START OF DATA PIPELINE -----
# Define dataset loading and preprocessing code

# Load the MIDI file
def midi_to_tensor(midi_file):
    notes = midi_file.instruments[0].notes # Grab notes only
    time_resolution = TIME_RESOLUTION  # Quantize

    # Create tensor for MIDI: shape of (num_time_steps, num_pitches)
    num_time_steps = int(midi_file.get_end_time() * time_resolution) + 1
    num_pitches = NUM_PITCHES  # Assuming MIDI standard pitch range
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

# ----- End of Data Pipeline -----
# ----- NETWORK ARCHITECTURE -----
# Define the LSTM SoloistModel architecture

class Soloist(nn.Module):

    '''
    # The simpler method
    def __init__(self, in_size, hid_size, layers, out_size):
        super(Soloist, self).__init__()
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=hid_size, num_layers=layers, batch_first=True)
        self.in_size = in_size
        self.hid_size = hid_size
        self.layers = layers
        self.out_size = out_size
        self.dropout = nn.Dropout(0.2)
        # self.linear = nn.Linear(256, n_vocab)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(self.dropout(x))
        return x
    '''

    def __init__(self, input_size, hidden_size, classes_num, layers=2):
        super(Soloist, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.classes = classes_num
        self.layers = layers
        self.sequence_enc = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.batch_norm_layer = nn.BatchNorm1d(hidden_size)
        self.lstm_layer = nn.LSTM(hidden_size, hidden_size, layers)
        self.fc_layer = nn.Linear(hidden_size, classes_num)

    def forward(self, ip_seqs, ip_seqs_len, hidden_state=None):

        seq_enc = self.seqence_enc(ip_seqs) # Encode input sequence
        seq_enc_rol = seq_enc.permute(1, 2, 0).contiguous()  # Permute
        seq_enc_norm = self.bn_layer(seq_enc_rol)  # Normalize
        seq_enc_norm_dropout = nn.Dropout(0.25)(seq_enc_norm)  # Apply dropout for regularization
        seq_enc_revert = seq_enc_norm_dropout.permute(2, 0, 1)  # Revert sequence order

        seq_packed = torch.nn.utils.rnn.pack_padded_sequence(seq_enc_revert, ip_seqs_len)  # pack sequences for lstm
        lstm_output, hidden_state = self.lstm_layer(seq_packed, hidden_state)  # pass sequences through lstm layer

        output_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output)  # unpack sequences

        output_norm = self.batch_norm_layer(lstm_output.permute(1, 2, 0).contiguous())  # normalize
        output_norm_dropout = nn.Dropout(0.1)(output_norm)  # apply dropout layer
        logits = self.fc_layer(output_norm_dropout.permute(2, 0, 1))  # pass through fully connected layer --> logits
        logits = logits.transpose(0, 1).contiguous()

        zero_one_logits = torch.stack((logits, 1 - logits), dim=3).contiguous()  # stack logits and reverse logits
        flattened_logits = zero_one_logits.view(-1, 2)  # flatten logits
        return flattened_logits, hidden_state


loss_function = nn.CrossEntropyLoss().cpu()
# The soloist is born
soloist = Soloist(input_size=NUM_PITCHES, hidden_size=HIDDEN_SIZE, classes_num=NUM_PITCHES).cpu()

