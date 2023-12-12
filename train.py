# -*- coding: utf-8 -*-
import os.path

import numpy as np
import torch
import torch.nn as nn
from model import Toymodel
from read_data import load_data
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from model import DEMASQ

def train_toy_model():
    epochs = 300
    data_train, data_val = load_data()
    model = Toymodel(768)
    optimizer = Adam(lr=0.001, params=model.parameters())
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.
    patience = 30
    for e in tqdm(range(epochs)):
        model.train()
        cnt = 0.
        cnt_val = 0.
        cnt_pat = 0
        total_train_loss = 0.0
        for i, (x,y) in enumerate(data_train):
            cnt += 1
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        print(f'epoch: {e}, train loss: {total_train_loss/cnt}')
        model.eval()
        total_acc = 0.
        with torch.no_grad():
            for i, (x, y) in enumerate(data_val):
                cnt_val += 1
                logits = model(x)
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                total_acc += accuracy_score(y, preds)
        val_acc = total_acc/cnt_val
        print(f"epoch: {e}, val_acc: {val_acc}")
        if val_acc > best_acc:
            cnt_pat = 0
            best_acc = val_acc
            torch.save(model.state_dict(),'save_models/toy_model_best.pt')
        else:
            cnt_pat += 1
        if cnt_pat >= patience:
            break

def train_DEM_model(EPOCHS, LR, IN_DIM, SAVE_PATH):
    epochs = EPOCHS
    SAVE_PATH = os.path.join(SAVE_PATH, 'DEM_model_best.pt')
    data_train, data_val = load_data()
    model = DEMASQ(in_dim=IN_DIM)
    optimizer = Adam(lr=LR, params=model.parameters())
    #criterion = nn.CrossEntropyLoss()
    best_acc = 0.
    patience = 30
    for e in tqdm(range(epochs)):
        model.train()
        cnt = 0.
        cnt_val = 0.
        cnt_pat = 0
        total_train_acc = 0.0
        for i, (x,y) in tqdm(enumerate(data_train)):
            cnt += 1
            optimizer.zero_grad()
            #logits = model(x)
            loss, preds = model.fit(x, y)
            #print(loss)
            loss.backward()
            optimizer.step()
            preds = preds.squeeze().detach().cpu().numpy()
            preds[preds > 0.5] = 1
            preds[preds < 0.5] = 0
            total_train_acc += accuracy_score(y, preds)
        print(f'epoch: {e}, train acc: {total_train_acc/cnt}')
        model.eval()
        total_acc = 0.
        with torch.no_grad():
            for i, (x, y) in enumerate(data_val):
                cnt_val += 1
                preds = model.inference(x)
                preds = preds.squeeze().detach().cpu().numpy()
                preds[preds > 0.5] = 1
                preds[preds < 0.5] = 0
                total_acc += accuracy_score(y, preds)
        val_acc = total_acc/cnt_val
        print(f"epoch: {e}, val_acc: {val_acc}")
        if val_acc > best_acc:
            cnt_pat = 0
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            cnt_pat += 1
        if cnt_pat >= patience:
            break

if __name__ == "__main__":
    train_DEM_model(1000, 0.0001, 768, 'save_models')