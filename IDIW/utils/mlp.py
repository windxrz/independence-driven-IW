import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from IDIW.utils.loss import LSIF_loss, weighted_cross_entropy, weighted_mse


class MLP(nn.Module):
    def __init__(self, shapes, acti="relu"):
        super(MLP, self).__init__()
        self.acti = acti
        self.fc = nn.ModuleList()
        for i in range(0, len(shapes) - 1):
            self.fc.append(nn.Linear(shapes[i], shapes[i + 1]))
        self.grad = True

    def forward(self, x):
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if i == len(self.fc) - 1:
                break
            if self.acti == "relu":
                x = F.relu(x)
            elif self.acti == "sigmoid":
                x = F.sigmoid(x)
            elif self.acti == "softplus":
                x = F.softplus(x)
            elif self.acti == "leakyrelu":
                x = F.leaky_relu(x)
        return x

    def freeze(self):
        for para in self.parameters():
            para.requires_grad = False
        self.grad = False

    def activate(self):
        for para in self.parameters():
            para.requires_grad = True
        self.grad = True


class MLPClassifier:
    def __init__(
        self,
        shapes,
        acti="relu",
        batch_size=256,
        lr=0.001,
        epoch=100,
        optim="Adam",
        model_dir=None,
        tensorboard_dir=None,
        loss="bce",
    ):
        self.mlp = MLP(shapes, acti)
        self.tensorboard_dir = tensorboard_dir
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.lr = lr
        self.optim = optim
        self.epoch = epoch
        self.loss = loss

    def fit(self, x, y, LSIF_lambda=0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = self.mlp.to(device)

        x = torch.from_numpy(x).type(torch.float)
        y = torch.from_numpy(y).view(-1, 1).type(torch.float)

        data = TensorDataset(x, y)

        sampler = RandomSampler(data)
        train_dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        optimizer = getattr(torch.optim, self.optim)(self.mlp.parameters(), lr=self.lr)

        min_l = 0
        min_epoch = 0

        for epoch_i in range(self.epoch):
            l = []
            for batch in train_dataloader:
                x = batch[0].to(device)
                y = batch[1].to(device)
                y_pred = self.mlp(x)
                if self.loss == "bce":
                    y_pred = torch.sigmoid(y_pred)
                    loss = weighted_cross_entropy(y, y_pred)
                elif self.loss == "LSIF":
                    y_pred = F.relu(y_pred)
                    loss = LSIF_loss(y, y_pred) + LSIF_lambda * torch.mean(
                        torch.square(y_pred)
                    )
                l.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch_i % 25 == 24:
                print(
                    "Epoch {:>3} / {}: loss {:.6f}".format(
                        epoch_i + 1,
                        self.epoch,
                        np.mean(l),
                    )
                )

            loss = np.mean(l)
            if epoch_i == 0 or loss < min_l:
                min_l = loss
                min_epoch = epoch_i

            if epoch_i - min_epoch > 50:
                break

    def predict(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = self.mlp.to(device)
        x = torch.from_numpy(x).type(torch.float).to(device)
        if self.loss == "bce":
            y = torch.sigmoid(self.mlp(x)).cpu().detach()
        else:
            y = F.relu(self.mlp(x)).cpu().detach()
        return y


class MLPRegressor:
    def __init__(
        self,
        shapes,
        batch_size=256,
        lr=0.001,
        epoch=1000,
        optim="Adam",
    ):
        self.mlp = MLP(shapes, "relu")
        self.batch_size = batch_size
        self.lr = lr
        self.optim = optim
        self.epoch = epoch

    def fit(self, X, y):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = self.mlp.to(device)

        X = torch.from_numpy(X).type(torch.float)
        y = torch.from_numpy(y).view(-1, 1).type(torch.float)

        data = TensorDataset(X, y)

        sampler = RandomSampler(data)
        train_dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        optimizer = getattr(torch.optim, self.optim)(self.mlp.parameters(), lr=self.lr)

        min_l = 0
        min_epoch = 0

        for epoch_i in range(self.epoch):
            l = []
            for batch in train_dataloader:
                X = batch[0].to(device)
                y = batch[1].to(device)
                y_pred = self.mlp(X)
                y_pred = torch.sigmoid(y_pred)
                loss = weighted_mse(y, y_pred)
                l.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch_i % 25 == 24:
                print(
                    "Epoch {:>3} / {}: loss {:.6f}".format(
                        epoch_i + 1,
                        self.epoch,
                        np.mean(l),
                    )
                )

            loss = np.mean(l)
            if epoch_i == 0 or loss < min_l:
                min_l = loss
                min_epoch = epoch_i

            if epoch_i - min_epoch > 50:
                break

    def predict(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = self.mlp.to(device)
        X = torch.from_numpy(X).type(torch.float).to(device)
        y = self.mlp(X).cpu().detach()
        return y
