#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import numpy as np
import pandas as pd
import torch
from torch import nn


# In[ ]:


seed = 4234343

torch.manual_seed(seed)
np.random.seed(seed)


# In[ ]:


np.set_printoptions(precision=4, floatmode='fixed', suppress=True)
pd.set_option({'display.precision':4})
torch.set_printoptions(precision=4, sci_mode=False)
device = "cpu"


# # LSTM Model

# In[ ]:


class LSTMAutoEncoder(nn.Module):
    
    def __init__(self, batch_size, sequence_length, input_size, hidden_size, latent_size, num_layers=1):
        super().__init__()
        
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_size = latent_size
        
        self.encode_hidden_state = []
        self.encode_cell_state = []

        # -------- Encoder (LSTMCell) --------
        
        self.encoder = nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=False,
            device=device
        )

        self.to_latent = nn.Linear(hidden_size, latent_size, bias=False, device=device)

        # -------- Decoder (LSTM) --------
        
        self.from_latent = nn.Linear(latent_size, hidden_size, bias=False, device=device)

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=False,
            batch_first=True,
            device=device
        )

        # -------- Output --------
        
        self.output_layer = nn.Linear(hidden_size, input_size,bias=False, device=device)

    
    def encode(self, x): # Encode entra em LSTMCell

        self.encode_hidden_state = []
            
        self.encode_cell_state = []

        for t in range(self.sequence_length):

            if t == 0:

                (h_t, c_t) = self.encoder(x[:,t,:]) # x: (batch, sequence_length = 1, input_size)
            
            else:
                
                h_prev = self.encode_hidden_state[-1]
                
                c_prev = self.encode_cell_state[-1]
                
                (h_t, c_t) = self.encoder(x[:,t,:], (h_prev, c_prev)) # (h_t, c_t): both are (batch, hidden_size)
                
            self.encode_hidden_state.append(h_t)
            self.encode_cell_state.append(c_t)
            
        z = self.to_latent(h_t) # (batch, hidden_size) @ (hidden_size, latent_size)

        return z # (batch, latent_size)
    
    def decode(self, z): # Decode entra em LSTM
        
        batch_size = z.size(0)

        h_0 = self.from_latent(z).unsqueeze(0).repeat(self.num_layers, 1, 1) # h_0: (num_layers, batch, hidden_size)
        
        c_0 = torch.zeros_like(h_0, device=device) # c_0: (num_layers, batch, hidden_size)

        decoder_x = torch.zeros(batch_size, self.sequence_length, self.hidden_size, device=device)

        out, _ = self.decoder(decoder_x, (h_0, c_0))

        x_hat = self.output_layer(out)

        return x_hat
    
    # -------- Forward --------
    def forward(self, x):
        
        z = self.encode(x)
            
        x_hat = self.decode(z)

        return x_hat