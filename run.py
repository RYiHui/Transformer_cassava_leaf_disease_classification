import os
import pandas as pd
import torch
from sklearn import model_selection
import CassavaDataset
import Global_Variable as gl
from torch.utils import data
import torch.nn as nn
import Model
from datetime import datetime
import Gpu


def run():
    df = pd.read_csv(os.path.join(gl.DATA_PATH, 'train.csv'))
    train_df, valid_df = model_selection.train_test_split(df, test_size=0.1, random_state=42, shuffle=True,
                                                         stratify=df.label.values)

    train_dataset = CassavaDataset.CassavaDataset(train_df, transforms=CassavaDataset.transforms_train)
    valid_dataset = CassavaDataset.CassavaDataset(valid_df,transforms=CassavaDataset.transforms_valid)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=gl.BATCH_SIZE,
        # sampler=train_sampler,
        drop_last=True,
        num_workers=8,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=gl.BATCH_SIZE,
        # sampler=valid_sampler,
        drop_last=True,
        num_workers=8,
    )

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = gl.LR
    model=Model.ViTBase16(n_classes=5, pretrained=True)
    model=model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #开始训练前
    start_time=datetime.now()
    logs = Gpu.fit_gpu(model=model,
                       epochs=gl.N_EPOCHS,
                       device=device,
                       criterion=criterion,
                       optimizer=optimizer,
                       train_loader=train_loader,
                       valid_loader=valid_loader)

    print(f"Execution time:{datetime.now() - start_time}")
    torch.save(model.state_dict(),f'end_model.pth')