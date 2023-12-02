import pandas as pd
import pytorch_lightning as pl

from lightning_classes import LinearModel, RatingsDataModule


def prepare_data_module():
    ratings = pd.read_csv("../data/processed/ratings_embeddings.csv")
    data_module = RatingsDataModule(ratings)

    return data_module


def evaluate():
    data_module = prepare_data_module()
    model = LinearModel.load_from_checkpoint("../models/linear_model.ckpt")

    trainer = pl.Trainer(accelerator="cpu")
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    evaluate()
