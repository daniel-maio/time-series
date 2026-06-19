#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import json

import joblib

import socket

import sklearn

import torch
from torch import nn

import sys
from pathlib import Path
root = Path.cwd().parent._str
sys.path.insert(0, root)

from scripts.funcs import get_anomaly_readings
from scripts.model_class import LSTMAutoEncoder


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


# In[ ]:


root = Path.cwd().parent
scaler_model_path = "scaler_model/scaler.joblib"
loss_tensor_path = "lstm_model/loss_tensor.pt"
lstm_model_path = "lstm_model/lstm_model.pt"


# In[ ]:


scaler_model = joblib.load(scaler_model_path)


# In[ ]:


loss_tensor = torch.load(loss_tensor_path)


# In[ ]:


file_pt = torch.load(lstm_model_path)

cfg = file_pt['config']

lstm_model = LSTMAutoEncoder(**cfg)

lstm_model.load_state_dict(file_pt["model_state"])


# In[ ]:


server = socket.socket()
server.bind(("127.0.0.1", 9999))
server.listen(1)

print("Waiting connection...")
conn, addr = server.accept()

print(f"Connected.")
print(f"Server Adress: {conn.getsockname()}")
print(f"Client Adress: {addr}")


# In[ ]:


while True:

    lstm_model.eval()

    data = conn.recv(100000)

    payload = json.loads(data.decode())

    if not data:
        break

    try:
        df = pd.DataFrame(payload)

        print(f"Incoming data...\n{df}\n")
        conn.send(f"Incoming data...\n{df}\n".encode())

        scaled_seq = scaler_model.transform(df)

        X = torch.tensor(scaled_seq.values, dtype=torch.float32, device=device).unsqueeze(0) # (1, sequence_length, input_size)

        print(f"Tensor data loaded with size: {X.size()}\n")
        conn.send(f"Tensor data loaded with size: {X.size()}\n".encode())

        loss_criterion = nn.MSELoss(reduction='none')

        with torch.no_grad():
            print(f"Inference started...\n")
            conn.send(b"Inference started...\n")

            x_hat = lstm_model(X)
            errors = loss_criterion(x_hat, X)

            print(f"Erros: {errors}\n")

            anomalies = get_anomaly_readings(loss_tensor, errors, df, std=3)

            if len(anomalies) > 1:
                msg = "\033[91m🚨 Anomalies detected!\033[0m"
                conn.send(msg.encode())
                conn.send(f"\n{anomalies}\n".encode())
                print(f"Readings with anomaly:\n{anomalies}\n")
                
    except Exception as e:
        print(e)

