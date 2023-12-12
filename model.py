# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os
import numpy as np
from captum.attr import IntegratedGradients
from scipy.special import jn_zeros
from copy import deepcopy

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["HF_DATASETS_CACHE"] = "E:/Data/torch_model"
# os.environ["HF_HOME"] = "E:/Data/torch_model"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "E:/Data/torch_model"
# os.environ["TRANSFORMERS_CACHE"] = "E:/Data/torch_model"

class Toymodel(nn.Module):
    def __init__(self, in_dim):
        super(Toymodel, self).__init__()
        self.lin1 = nn.Linear(in_dim, 256)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(64, 2)

    def forward(self, x):
        return self.lin3(self.relu2(self.lin2(self.relu1(self.lin1(x)))))


class Embeddings_Extractor:
    def __init__(self):
        """
        I use r"transformer" because I download the files and use the model locally.
        If you want to load weights from huggingFace automatically,
        use the codes commented out, "sentence-transformers/msmarco-distilbert-base-tas-b".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(r"transformer")
        self.model = AutoModel.from_pretrained(r"transformer")
        # self.tokenizer = AutoTokenizer.from_pretrained(r"sentence-transformers/msmarco-distilbert-base-tas-b")
        # self.model = AutoModel.from_pretrained(r"sentence-transformers/msmarco-distilbert-base-tas-b")

    def avg_pooling(self, x):
        return torch.mean(x.last_hidden_state, dim=1)

    def encode(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)

        # Perform pooling
        embeddings = self.avg_pooling(model_output)

        return embeddings


class IG_block:
    def __init__(self):
        super(IG_block, self).__init__()
        self.model = Toymodel(768)
        self.model.load_state_dict(torch.load('save_models/toy_model_best.pt'), strict=False)
        self.IG = IntegratedGradients(self.model)

    def inverse(self, embeds, tar, max_features=20):
        baseline = torch.zeros_like(embeds)
        ig_attrs, _ = self.IG.attribute(inputs=embeds, baselines=baseline, target=tar, n_steps=200,
                                        return_convergence_delta=True)
        max_ids = torch.argsort(torch.abs(ig_attrs), dim=1)[:, -max_features:].detach().cpu().numpy().tolist()
        B_size = embeds.shape[0]
        H_size = embeds.shape[1]
        embeds = embeds.expand(20, B_size, H_size)
        embeds = embeds.permute(1, 0, 2)
        #print(B_size)
        for i in range(B_size):
            for j in range(max_features):
                idx = max_ids[i][j]
                indices = (torch.LongTensor([i]), torch.LongTensor([j]), torch.LongTensor([idx]))
                embeds = deepcopy(embeds.index_put(indices, torch.Tensor([0.0])))
        return embeds


class LinearModule(nn.Module):
    def __init__(self, in_dim):
        """
        sigmoid function is important!
        :param in_dim: input dim of text embedding
        """
        super(LinearModule, self).__init__()
        n_hidden = [512, 256, 128, 64, 32, 1]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, n_hidden[0]))
        self.layers.append(nn.ReLU())
        for i in range(1, len(n_hidden) - 1):
            self.layers.append(nn.Linear(n_hidden[i - 1], n_hidden[i]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(32, n_hidden[-1]))
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out_act(x)
        return x


class Energy_part(nn.Module):
    def __init__(self):
        super(Energy_part, self).__init__()
        self.VS = 0.8

    def cal_zero(self, x_unique: int):
        """
        calculate source zero frequency
        """
        d_nodes = 0  #diametric nodes
        c_nodes = x_unique #circular nodes
        if d_nodes == 0:
            J01 = 2.404825557695773
        else:
            J01 = jn_zeros(d_nodes, 1)[0]
        if x_unique == 768:
           J0m = 2411.957811618678
        else:
           J0m = jn_zeros(d_nodes, c_nodes)[-1]
        #print(jn_zeros(d_nodes, c_nodes))
        Drumhead_frequency = torch.from_numpy(np.array([J0m / J01]))
        #print(Drumhead_frequency)
        return Drumhead_frequency

    def source_fre(self, x):
        """
        calculate source frequency
        :param x:  input tensor
        :return: source frequency
        """
        min_val = torch.min(x, dim=2, keepdim=True)[0]
        max_val = torch.max(x, dim=2, keepdim=True)[0]
        neg_indices = torch.where(min_val < 0)
        pos_indices = torch.where(min_val > 0)
        neg_indices = (neg_indices[0].detach().cpu().numpy().tolist(), neg_indices[1].detach().cpu().numpy().tolist())
        pos_indices = (pos_indices[0].detach().cpu().numpy().tolist(), pos_indices[1].detach().cpu().numpy().tolist())
        for i in range(len(neg_indices[0])):
            x[neg_indices[0][i], neg_indices[1][i], :] += min_val[neg_indices[0][i], neg_indices[1][i], :]
        for i in range(len(pos_indices[0])):
            x[pos_indices[0][i], pos_indices[1][i], :] -= max_val[pos_indices[0][i], pos_indices[1][i], :]
        """
        if min_val < 0:
            x += min_val
        else:
            x -= max_val
        """
        FRE = torch.zeros_like(x[..., :1])
        B_size = x.shape[0]
        R_size = x.shape[1]
        #x += min_val
        for i in range(B_size):
            for j in range(R_size):
                x_unique = torch.unique(x[i, j, :])
                num_uni = len(x_unique)
                zero_point = self.cal_zero(num_uni)
                FRE[i, j, :] = deepcopy(zero_point)
                #print(zero_point.all() == FRE[i, j, :].all())
        return FRE

    def forward(self, x, labels):
        B_size = x.shape[0]
        R_size = x.shape[1]
        C = torch.var(x, dim=-1)
        VR = torch.mul(labels, torch.abs(C))
        Ef0 = self.source_fre(x)
        dover_fun = torch.div((C + VR), (C - self.VS) + 1e-6).reshape([B_size, R_size, 1])
        Ef = dover_fun * Ef0
        if len(x.shape) == 3:
            Ef = torch.mean(Ef, dim=1)
        return Ef

class DEMASQ(nn.Module):
    def __init__(self, in_dim):
        super(DEMASQ, self).__init__()
        self.net = LinearModule(in_dim)
        self.IG = IG_block()
        self.energy_net = Energy_part()
        self.criterion = nn.BCELoss()

    def forward(self, x):
        out = self.net(x)
        return out

    def fit(self, x, y):
        B_size = x.shape[0]
        preds = self.forward(x)
        y = y.reshape([-1, 1])
        #print(preds)
        BCE_loss = self.criterion(preds, y.float())
        x_permuted = self.IG.inverse(x, y.squeeze().long())
        energy_loss = self.energy_net(x_permuted, y)
        ener_0 = self.energy_net(x_permuted, torch.zeros((B_size, 1)))
        ener_1 = self.energy_net(x_permuted, torch.ones((B_size, 1)))
        total_loss = torch.mean(BCE_loss + energy_loss - torch.minimum(ener_0, ener_1))
        return total_loss, preds

    def inference(self, x):
        preds = self.net(x)
        return preds

if __name__ == "__main__":
    # X = torch.rand((100,100))
    # model = LinearModule(100)
    # print(X)
    # print(model(X))
    #sentence = ["Where are you", "How are you"]
    #embed_catcher = Embeddings_Extractor()
    #rs = embed_catcher.encode(sentence)
    #IG = IG_block()
    #labels = torch.LongTensor([1, 1])
    #inv_res = IG.inverse(rs, labels)
    #cri = nn.BCELoss()
    #model = DEMASQ(in_dim=768)
    #print(model.fit(rs, labels))
    print(jn_zeros(0, 1)[-1])
    #print(torch.sum(inv_res == 0))
    #model = Energy_part()
    #print(model(inv_res, labels).shape)
    #print(torch.sum(inv_res == 0))