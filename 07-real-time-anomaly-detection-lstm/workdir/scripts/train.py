#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import numpy as np
import pandas as pd

import joblib

import sklearn
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


# In[ ]:


seed = 4234343

torch.manual_seed(seed)
np.random.seed(seed)


# In[ ]:


np.set_printoptions(precision=4, floatmode='fixed', suppress=True)
pd.set_option({'display.precision':4})
sklearn.set_config(transform_output="pandas")
torch.set_printoptions(precision=4, sci_mode=False)
device = "cpu"


# # Objective

# > Anomaly detection in non-supervisioned data

# # Paths

# In[ ]:


from pathlib import Path
root = Path.cwd().parent


# In[ ]:


csv_file_path = "data/dataset.csv"

scaler_model_path = "scaler_model/scaler.joblib"

loss_tensor_path = "lstm_model/loss_tensor.pt"
lstm_model_path = "lstm_model/lstm_model.pt"


# # Load Data

# In[ ]:


df = pd.read_csv(csv_file_path, header=0, parse_dates=True, index_col=1)


# In[ ]:


df.head()


# In[ ]:


df.info()


# > Remove column "sensor_id"

# In[ ]:


df.drop(labels='sensor_id', axis=1, inplace=True)


# In[ ]:


df.describe()


# In[ ]:


df.head(10)


# # Pre-processing

# > Set time sequence

# In[ ]:


df.sort_index(inplace=True)


# > Check if time series is regular (regular intervals)

# In[ ]:


df.index.to_series().diff().describe()


# > Time sequence is not regular, but we will ignore it

# # Split Data

# In[ ]:


n = int(len(df) * 0.8)


# In[ ]:


train = df[:n]
test = df[n:]


# # Normalize

# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaler.fit(train)


# In[ ]:


joblib.dump(scaler, scaler_model_path)


# In[ ]:


train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)


# # Create Time Window

# > (batch, sequence_length, features)

# In[ ]:


def create_sequences(df, seq_len):
    X = []

    for i in range(len(df) - seq_len):
        X.append(df[i:i+seq_len].values.tolist())

    X = torch.tensor(X, dtype=torch.float32, device=device)
    print(X.shape)

    return X


# In[ ]:


sequence_length = 5 # timesteps

train_seq = create_sequences(train_scaled, sequence_length)
test_seq = create_sequences(test_scaled, sequence_length)


# In[ ]:


train_scaled.shape, test_scaled.shape


# In[ ]:


train_seq.shape, test_seq.shape


# # Dataloader

# In[ ]:


batch_size = 15


# In[ ]:


train_dataset = TensorDataset(train_seq)
#test_dataset = TensorDataset(test_seq)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[ ]:


(next(iter((train_loader)))[0]).size()


# In[ ]:


batch_size, sequence_length, input_size = (next(iter((train_loader)))[0]).size()
batch_size, sequence_length, input_size


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


# # Training

# In[ ]:


# x: (batch_size, sequence_length, input_size)

batch_size, sequence_length, input_size


# In[ ]:


epochs = 1051
sequence_length = sequence_length
input_size = input_size
hidden_size = 8
latent_size = 4
lr = 0.1


# In[ ]:


lstm_model = LSTMAutoEncoder(batch_size, sequence_length, input_size, hidden_size, latent_size)


# In[ ]:


loss_criterion = nn.MSELoss(reduction='none')
optimizer = Adam(lstm_model.parameters(), lr=lr)


# In[ ]:


lstm_model.train()

for epoch in range(epochs):

    for x in train_loader:

        # forward
        x_hat = lstm_model(x[0])

        loss_vector = loss_criterion(x_hat, x[0])
        loss = loss_vector.mean()

        # backprop
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}\nLoss: {loss_vector}\nMSE: {loss.item()}")
            print(f"---------")


# In[ ]:


torch.save(loss_vector, loss_tensor_path)


# # Detect Anomaly

# ## Test Dataset

# In[ ]:


lstm_model.eval()
with torch.no_grad():
    x_hat_test = lstm_model(test_seq)

x_hat_test.shape


# In[ ]:


test_errors = loss_criterion(x_hat_test, test_seq)
test_errors.shape


# In[ ]:


def get_anomaly_readings(errors, original_df, std = 2):

    if len(errors.size()) < 3:
        batch_size = 1
    else:
        batch_size = errors.size(0)

    std_ = torch.std(loss_vector, dim=0).unsqueeze(0).repeat(batch_size,1,1)

    anomaly_mask = torch.sqrt(errors) > std * std_

    i, j = torch.nonzero(anomaly_mask, as_tuple=True)[:-1]

    idx = torch.unique(i + j)

    return original_df.iloc[idx]


# > Readings with anomaly in the test sequence

# In[ ]:


test_anomalies = get_anomaly_readings(test_errors, test_scaled)
print(test_anomalies)


# ## New Dataset

# In[ ]:


df.describe()


# In[ ]:


new_x = pd.DataFrame(
    data=
            {
                "temperature": np.random.randint(low=16,high=24,size=(sequence_length,)),
                "humidity": np.random.randint(low=32,high=65,size=(sequence_length,)),
                "pressure": np.random.randint(low=1009,high=1020,size=(sequence_length,))
            }
)


# In[ ]:


print(new_x)


# In[ ]:


new_x_scaled = scaler.transform(new_x)
print(new_x_scaled)


# In[ ]:


new_sequence = torch.tensor(new_x_scaled.to_numpy(), dtype=torch.float32).reshape(1, sequence_length, input_size)
print(new_sequence)


# In[ ]:


lstm_model.eval()
with torch.no_grad():

    new_x_hat = lstm_model(new_sequence)
    new_errors = loss_criterion(new_x_hat, new_sequence)

new_errors


# > Readings with anomaly of the new dataset

# In[ ]:


new_anomaly_readings = get_anomaly_readings(new_errors, new_x)
print(new_anomaly_readings)


# In[ ]:


# Comparing:

print(new_x_scaled)


# # Save Model

# In[ ]:


config = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "latent_size": latent_size,
            "num_layers":1
        }


# In[ ]:


torch.save(
    obj=
        {
            "model_state": lstm_model.state_dict(),
            "config": config
        },
    f=lstm_model_path
)


# In[ ]:


print(f"Model saved.")
print(f"--------")
print("End of script")

