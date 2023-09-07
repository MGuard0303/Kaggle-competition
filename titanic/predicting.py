import datetime
import pathlib

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import logic
import Model


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


df_path = pathlib.Path('./datasets/test_vanilla.csv')
para_path = pathlib.Path("./paras/20230907_160139.pt")

df = pd.read_csv(df_path, index_col=0).reset_index(drop=True)
features = torch.tensor([df.Pclass, df.Sex, df.Age], dtype=torch.float).transpose(0, 1).to(device)

dataset = TensorDataset(features)
dataloader = DataLoader(dataset, batch_size=len(features))

model = Model.Vanilla().to(device)
model.load_state_dict(torch.load(para_path))

predictions = logic.predict(model, dataloader)
predictions = pd.Series(predictions)

current = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result_path = pathlib.Path("./results/" + f"{current}.csv")
predictions.to_csv(result_path, index=False)
