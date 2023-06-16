import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import(
    SEG_NUM_TIMESTEPS
)
class FAT(nn.Module):
    def __init__(self, num_ifos, num_timesteps, BOTTLENECK, FACTOR):
        super(FAT, self).__init__()
        print("WARNING: Change this with Eric's actual LSTM model!")
        self.num_timesteps = num_timesteps
        self.num_ifos = num_ifos
        self.Linear1 = nn.Linear(num_timesteps*2, 2**7)
        self.Linear2 = nn.Linear(2**7, 2**11)
        self.Linear3 = nn.Linear(2**11, BOTTLENECK)
        self.Linear4 = nn.Linear(BOTTLENECK, 2**11)
        self.Linear5 = nn.Linear(2**11, 2**7)
        self.Linear6 = nn.Linear(2**7, num_timesteps*2)
        #self.Conv2 = nn.Conv1d(in_channels=5, out_channels = 2, kernel_size = 5, padding = 'same')
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, self.num_timesteps*2)
        x = F.tanh(self.Linear1(x))
        x = F.tanh(self.Linear2(x))
        x = F.tanh(self.Linear3(x))
        x = F.tanh(self.Linear4(x))
        x = F.tanh(self.Linear5(x))
        x = (self.Linear6(x))
        x = x.view(batch_size, 2, self.num_timesteps)
        #x = self.Conv2(x)
        return x


class DUMMY_CNN_AE(nn.Module):
    def __init__(self, num_ifos, num_timesteps, BOTTLENECK, FACTOR):
        super(DUMMY_CNN_AE, self).__init__()
        print("WARNING: Change this with Eric's actual LSTM model!")
        self.num_timesteps = num_timesteps
        self.num_ifos = num_ifos
        self.Conv1 = nn.Conv1d(in_channels = num_ifos, out_channels = 5, kernel_size=5, padding='same')
        self.Linear1 = nn.Linear(num_timesteps*5, SEG_NUM_TIMESTEPS)
        self.Linear2 = nn.Linear(SEG_NUM_TIMESTEPS, BOTTLENECK)
        self.Linear3 = nn.Linear(BOTTLENECK, num_timesteps*5)
        self.Conv2 = nn.Conv1d(in_channels=5, out_channels = 2, kernel_size = 5, padding = 'same')
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.Conv1(x))
        x = x.view(-1, self.num_timesteps*5)
        x = F.tanh(self.Linear1(x))
        x = F.tanh(self.Linear2(x))
        x = F.tanh(self.Linear3(x))
        x = x.view(batch_size, 5, self.num_timesteps)
        x = self.Conv2(x)
        return x


# Third Party
import torch
import torch.nn as nn


class Encoder_LSTM_AE(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ, out_activ):
        super(Encoder_LSTM_AE, self).__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ, self.out_activ = h_activ, out_activ

    def forward(self, x):
        x = x.unsqueeze(0)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
            elif self.out_activ and index == self.num_layers - 1:
                return self.out_activ(h_n).squeeze()

        return h_n.squeeze()


class Decoder_LSTM_AE(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ):
        super(Decoder_LSTM_AE, self).__init__()

        layer_dims = [input_dim] + h_dims + [h_dims[-1]]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ = h_activ
        self.dense_matrix = nn.Parameter(
            torch.rand((layer_dims[-1], out_dim), dtype=torch.float), requires_grad=True
        )

    def forward(self, x, seq_len):
        x = x.repeat(seq_len, 1).unsqueeze(0)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)

        return torch.mm(x.squeeze(0), self.dense_matrix)

class LSTM_AE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        h_dims=[],
        h_activ=nn.Sigmoid(),
        out_activ=nn.Tanh(),
    ):
        super(LSTM_AE, self).__init__()

        self.encoder = Encoder_LSTM_AE(input_dim, encoding_dim, h_dims, h_activ, out_activ)
        self.decoder = Decoder_LSTM_AE(encoding_dim, input_dim, h_dims[::-1], h_activ)

    def forward(self, x):
        seq_len = x.shape[0]
        x = self.encoder(x)
        x = self.decoder(x, seq_len)

        return x
    
import torch.nn as nn
import torch.nn.functional as F
from torch import transpose as torchtranspose
class Encoder_ERIC(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder_ERIC, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
  def forward(self, x):
    batch_size = x.shape[0]
    x = x.reshape((batch_size, self.seq_len, self.n_features))
    print("A", x.shape)
    x, (_, _) = self.rnn1(x)
    print("B", x.shape)
    x, (hidden_n, _) = self.rnn2(x)
    print("C", x.shape)
    return hidden_n.reshape((batch_size, self.embedding_dim))
  
class Decoder_ERIC(nn.Module):
  def __init__(self, seq_len, n_features=1, input_dim=64,):
    super(Decoder_ERIC, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  def forward(self, x):
    batch_size = x.shape[0]
    x = x.unsqueeze(1)
    x = x.repeat(1, self.seq_len, 1)
    print("172", x.shape)
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((batch_size, self.seq_len, self.hidden_dim))
    return self.output_layer(x)
  
class LSTM_AE_ERIC(nn.Module):
  def __init__(self, num_ifos, num_timesteps, BOTTLENECK, FACTOR):
    super(LSTM_AE_ERIC, self).__init__()
    print("WARNING: This is LITERALLY Eric's model!!!")
    for i in range(50): 
       continue
       print("MINECRAFT MINECRAFT MINECRAFT")
    self.num_timesteps = num_timesteps
    self.num_ifos = num_ifos
    self.BOTTLENECK = BOTTLENECK
    self.encoder = Encoder_ERIC(seq_len=num_timesteps, n_features=num_ifos, embedding_dim=BOTTLENECK)
    self.decoder = Decoder_ERIC(seq_len=num_timesteps, n_features=num_ifos, input_dim=BOTTLENECK)
  def forward(self, x):
    #x = torchtranspose(x, 1, 2)
    x = self.encoder(x)
    x = self.decoder(x)
    x = torchtranspose(x, 1, 2)
    return x