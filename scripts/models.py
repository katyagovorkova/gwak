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
        self.Linear1 = nn.Linear(num_timesteps*2, 2**9)
        self.Linear2 = nn.Linear(2**9, 2**11)
        self.Linear3 = nn.Linear(2**11, BOTTLENECK)
        self.Linear4 = nn.Linear(BOTTLENECK, 2**11)
        self.Linear5 = nn.Linear(2**11, 2**9)
        self.Linear6 = nn.Linear(2**9, num_timesteps*2)
        #self.Conv2 = nn.Conv1d(in_channels=5, out_channels = 2, kernel_size = 5, padding = 'same')
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, self.num_timesteps*self.num_ifos)
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))
        x = F.relu(self.Linear4(x))
        x = F.tanh(self.Linear5(x))
        x = (self.Linear6(x))
        x = x.view(batch_size, self.num_ifos, self.num_timesteps)
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

class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 4 * embedding_dim 
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
    self.linear0 = nn.Linear(in_features=self.embedding_dim*seq_len, out_features=self.hidden_dim*10)
    self.linear1 = nn.Linear(in_features=self.hidden_dim*10, out_features=self.hidden_dim*5)
    self.linear2 = nn.Linear(in_features=self.hidden_dim*5, out_features=self.hidden_dim*2)
    self.linear3 = nn.Linear(in_features=self.hidden_dim*2, out_features=self.embedding_dim)

  def forward(self, x):
    batch_size = x.shape[0]
    #x = x.reshape((batch_size, self.seq_len, self.n_features))
    x, (hidden_n, cell_n) = self.rnn1(x)
    #print("92", hidden_n.shape, cell_n.shape)
    #print("x", x.shape)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = self.linear0(x.reshape(batch_size, -1))
    x = F.tanh(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = F.relu(self.linear3(x))
    #return hidden_n.reshape((batch_size, self.embedding_dim)) #traditional way
    return x.reshape((batch_size, self.embedding_dim)) #phil harris way

class Decoder(nn.Module):
  def __init__(self, seq_len, n_features=1, input_dim=64,):
    super(Decoder, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = input_dim, n_features
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
    self.linear1 = nn.Linear(self.hidden_dim, 2**9)
    self.linear2 = nn.Linear(2**9, self.hidden_dim*self.seq_len)

    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  def forward(self, x):
    batch_size = x.shape[0]
    #print("x.shape", x.shape)
    if 0:
        print("121", x.shape)
        x = x.unsqueeze(1)
        print("123", x.shape)
        #print("unsqueezes.shape", x.shape)
        x = x.repeat(1, self.seq_len, 1)
        print("after repeat", x.shape)
        #print("repeats.shape", x.shape)
    x = F.relu(self.linear1(x))
    x = F.tanh(self.linear2(x))
    x = x.reshape(batch_size, self.seq_len, self.hidden_dim)
    #print("135", x.shape)
    #x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    #print("138", x.shape)
    #x = x.reshape((batch_size, self.seq_len, self.hidden_dim))
    return self.output_layer(x)

class LSTM_AE(nn.Module):
  def __init__(self, num_ifos, num_timesteps, BOTTLENECK, FACTOR):
    super(LSTM_AE, self).__init__()
    print("WARNING: This is LITERALLY Eric's model!!!")
    for i in range(50):
       continue
       print("MINECRAFT MINECRAFT MINECRAFT")
    self.num_timesteps = num_timesteps
    self.num_ifos = num_ifos
    self.BOTTLENECK = BOTTLENECK
    self.encoder = Encoder(seq_len=num_timesteps, n_features=num_ifos, embedding_dim=BOTTLENECK)
    self.decoder = Decoder(seq_len=num_timesteps, n_features=num_ifos, input_dim=BOTTLENECK)
    #self.encoder = Encoder(seq_len=num_ifos, n_features=num_timesteps, embedding_dim=BOTTLENECK)
    #self.decoder = Decoder(seq_len=num_ifos, n_features=num_timesteps, input_dim=BOTTLENECK)
  def forward(self, x):
    x = x.transpose(1, 2)
    x = self.encoder(x)
    x = self.decoder(x)
    x = x.transpose(1, 2)
    return x


#model which splits and uses separate LSTMs for each detecor channel
#model which splits and uses separate LSTMs for each detecor channel
#torch.manual_seed(42)
class Encoder_SPLIT(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder_SPLIT, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, embedding_dim // 2
    self.rnn1_0 = nn.LSTM(
      input_size=1,
      hidden_size=self.hidden_dim//2,
      num_layers=1,
      batch_first=True
    )
    self.rnn1_1 = nn.LSTM(
      input_size=1,
      hidden_size=self.hidden_dim//2,
      num_layers=1,
      batch_first=True
    )

    self.rnn2_0 = nn.LSTM(
      input_size=self.hidden_dim//2,
      hidden_size=embedding_dim//2,
      num_layers=1,
      batch_first=True
    )
    self.rnn2_1 = nn.LSTM(
      input_size=self.hidden_dim//2,
      hidden_size=embedding_dim//2,
      num_layers=1,
      batch_first=True
    )
    #self.linear0 = nn.Linear(in_features=2**9, out_features=self.hidden_dim*10)
    self.linear1 = nn.Linear(in_features=2**9, out_features=self.hidden_dim*5)
    self.linear2 = nn.Linear(in_features=self.hidden_dim*5, out_features=self.hidden_dim*2)
    self.linear3 = nn.Linear(in_features=self.hidden_dim*2, out_features=self.embedding_dim)

    self.linearH = nn.Linear(embedding_dim//2 * seq_len, 2**8)
    self.linearL = nn.Linear(embedding_dim//2 * seq_len, 2**8)
  def forward(self, x):
    batch_size = x.shape[0]
    Hx, Lx = x[:, :, 0][:, :, None], x[:, :, 1][:, :, None] 

    Hx, (_, _) = self.rnn1_0(Hx)
    Hx, (_, _) = self.rnn2_0(Hx)
    Hx = Hx.reshape(batch_size, self.embedding_dim//2*self.seq_len)
    Hx = F.tanh(self.linearH(Hx))

    Lx, (_, _) = self.rnn1_1(Lx)
    Lx, (_, _) = self.rnn2_1(Lx)
    Lx = Lx.reshape(batch_size, self.embedding_dim//2*self.seq_len)
    Lx = F.tanh(self.linearL(Lx))

    x = torch.cat([Hx, Lx], dim=1)
    x = F.tanh(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = F.relu(self.linear3(x))

    return x.reshape((batch_size, self.embedding_dim)) #phil harris way

class Decoder_SPLIT(nn.Module):
  def __init__(self, seq_len, n_features=1, input_dim=64,):
    super(Decoder_SPLIT, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = input_dim, n_features
    self.rnn1_0 = nn.LSTM(
      input_size=1,
      hidden_size=1,
      num_layers=1,
      batch_first=True
    )
    self.rnn1_1 = nn.LSTM(
      input_size=1,
      hidden_size=1,
      num_layers=1,
      batch_first=True
    )
    self.linearH = nn.Linear(self.seq_len, self.seq_len)
    self.linearL = nn.Linear(self.seq_len, self.seq_len)
    self.linear1 = nn.Linear(self.hidden_dim, 2**9)
    self.linear2 = nn.Linear(2**9, self.seq_len)

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

    self.linearH_2 = nn.Linear(self.seq_len, self.seq_len)
    self.linearL_2 = nn.Linear(self.seq_len, self.seq_len)
  def forward(self, x):
    batch_size = x.shape[0]
    
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))

    Hx = self.linearH(x)[:, :, None]
    Lx = self.linearL(x)[:, :, None]

    x = torch.cat([Hx, Lx], dim=2)
    Hx, (_, _) = self.rnn1_0(Hx)
    Hx = self.linearH_2(Hx[:, :, 0])[:, :, None]

    Lx, (_, _) = self.rnn1_1(Lx)
    Lx = self.linearL_2(Lx[:, :, 0])[:, :, None]
    
    x = torch.cat([Hx, Lx], dim=2)


    return (x)

class LSTM_AE_SPLIT(nn.Module):
  def __init__(self, num_ifos, num_timesteps, BOTTLENECK, FACTOR):
    super(LSTM_AE_SPLIT, self).__init__()
    print("WARNING: This is LITERALLY Eric's model!!!")
    for i in range(50):
       continue
       print("MINECRAFT MINECRAFT MINECRAFT")
    self.num_timesteps = num_timesteps
    self.num_ifos = num_ifos
    self.BOTTLENECK = BOTTLENECK
    self.encoder = Encoder_SPLIT(seq_len=num_timesteps, n_features=num_ifos, embedding_dim=BOTTLENECK)
    self.decoder = Decoder_SPLIT(seq_len=num_timesteps, n_features=num_ifos, input_dim=BOTTLENECK)
    #self.encoder = Encoder(seq_len=num_ifos, n_features=num_timesteps, embedding_dim=BOTTLENECK)
    #self.decoder = Decoder(seq_len=num_ifos, n_features=num_timesteps, input_dim=BOTTLENECK)
  def forward(self, x):
    x = x.transpose(1, 2)
    #print("encoder")
    x = self.encoder(x)
    
    #(a(x))
    #print("decoder")
    x = self.decoder(x)
    
    #(a(x))
    x = x.transpose(1, 2)
    return x
