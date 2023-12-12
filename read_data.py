# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import Embeddings_Extractor
from tqdm import tqdm
import os


def load_data():
    """
    in this article, 0 for chatgpt and 1 for human beings
    :return:
    """
    df1 = pd.read_csv('dataset/train_essays.csv')
    df2 = pd.read_csv('dataset/train_v2_drcat_02.csv')
    texts_total = df1['text'].tolist() + df2['text'].tolist()
    lenth = len(texts_total)
    y_total = np.concatenate((df1['generated'].values, df2['label'].values))
    zero_idxes = np.where(y_total == 0)[0].tolist()
    one_idxes = np.where(y_total == 1)[0].tolist()
    y_total[zero_idxes] = 1
    y_total[one_idxes] = 0

    # y_total = y_total[:lenth]
    X_total = []
    embed_catcher = Embeddings_Extractor()
    # X_total = embed_catcher.encode(texts_total)
    if os.path.isfile(r'save_feat/feat.pth'):
        X_total = torch.load(r'save_feat/feat.pth')
    else:
        for i in tqdm(range(lenth)):
            X_total.append(embed_catcher.encode(texts_total[i]))
        X_total = torch.cat(X_total, dim=0)
        torch.save(X_total, 'save_feat/feat.pth')
    print(X_total.shape)
    train_dim = int(X_total.shape[0] * 0.7)
    y_total = torch.from_numpy(y_total)
    X_train = X_total[:train_dim, :]
    y_train = y_total[:train_dim]
    X_val = X_total[train_dim:, :]
    y_val = y_total[train_dim:]
    train_set = TensorDataset(X_train, y_train)
    val_set = TensorDataset(X_val, y_val)
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=32, shuffle=False)
    return train_loader, val_loader


if __name__ == "__main__":
    load_data()
