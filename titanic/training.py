import datetime
import pathlib

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import logic
import Model


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_avaliable():
    device = "mps"
else:
    device = "cpu"


df_path = pathlib.Path("./datasets/train_vanilla.csv")

df = pd.read_csv(df_path, index_col=0)
df = df.sample(frac=1.0).reset_index(drop=True)
features = torch.tensor([df.Pclass, df.Sex, df.Age], dtype=torch.float).transpose(0, 1).to(device)
labels = torch.tensor(df.Survived, dtype=torch.float).unsqueeze(1).to(device)

dataset = TensorDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Model.Vanilla().to(device)
model.optimizer = torch.optim.Adam(model.parameters())

mdl = logic.train(model, dataloader)

current = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
para_path = pathlib.Path("./paras/" + f"{current}.pt")
torch.save(mdl.state_dict(), para_path)
