import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd


class RatingsDataset(Dataset):
    def __init__(self, ratings):
        self.ratings = ratings.sort_values(by="timestamp")
        self.X = ratings.drop(columns=["rating", "user_id", "item_id", "timestamp"]).values
        self.y = ratings["rating"].values

    def split_by_timestamp(self, test_size=0.2):
        train = []
        test = []

        for user_id in self.ratings["user_id"].unique():
            user_ratings = self.ratings[self.ratings["user_id"] == user_id]
            cutoff = int((1 - test_size) * len(user_ratings))

            train_ratings = user_ratings[:cutoff]
            test_ratings = user_ratings[cutoff:]

            train.append(train_ratings)
            test.append(test_ratings)

        train = pd.concat(train)
        test = pd.concat(test)

        return RatingsDataset(train), RatingsDataset(test)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        user_x = self.X[idx][:29]
        movie_x = self.X[idx][29:54]
        movie_emb = self.X[idx][54:]

        return torch.tensor(user_x, dtype=torch.float32), torch.tensor(movie_x, dtype=torch.float32), torch.tensor(
            movie_emb, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


class RatingsDataModule(pl.LightningDataModule):
    def __init__(self, ratings, batch_size=512):
        super().__init__()
        self.ratings = ratings
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        self.dataset = RatingsDataset(self.ratings)

    def setup(self, stage: str = None) -> None:
        self.train, self.test = self.dataset.split_by_timestamp(test_size=0.2)
        self.train, self.val = self.train.split_by_timestamp(test_size=0.1)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class LinearModel(pl.LightningModule):
    def __init__(self, user_dim, movie_dim, emb_dim, output_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr

        self.user_dim = user_dim
        self.movie_dim = movie_dim
        self.emb_dim = emb_dim
        self.output_dim = output_dim

        self.movie_emb_linear = nn.Linear(self.emb_dim, 128)

        self.final_linear = nn.Linear(128 + 54, self.output_dim)

        self.batchnorm_emb = nn.BatchNorm1d(self.emb_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        user_x, movie_x, movie_emb = x

        movie_emb = self.batchnorm_emb(movie_emb)
        movie_emb = F.relu(self.movie_emb_linear(movie_emb))
        movie_emb = self.dropout(movie_emb)

        output = self.final_linear(torch.cat((user_x, movie_x, movie_emb), dim=1))

        return output

    def training_step(self, batch, batch_idx):
        x = batch[:-1]
        y = batch[-1]

        y_hat = self(x)

        loss = F.mse_loss(y_hat, y.unsqueeze(1))

        self.log_dict({"train_loss": loss ** 0.5}, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[:-1]
        y = batch[-1]

        y_hat = self(x)

        loss = F.mse_loss(y_hat, y.unsqueeze(1))

        self.log_dict({"val_loss": loss ** 0.5}, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x = batch[:-1]
        y = batch[-1]

        y_hat = self(x)

        loss = F.mse_loss(y_hat, y.unsqueeze(1))

        self.log_dict({"test_loss": loss ** 0.5}, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
