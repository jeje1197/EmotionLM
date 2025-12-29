import torch
import pandas as pd
import numpy as np

from training.model import ALSModel

CSV_FILE_PATH = "./data/03_CORE_training_dataset_FINAL.csv"

def load_dataset():
    return pd.read_csv(CSV_FILE_PATH)

def analyze_data(df):
    print(df.head())
    print(df.describe())

def create_model():
    return ALSModel()

def prepare_features(df, story_label=None):
    features = torch.tensor(df[["Semantic_Similarity", "Emotional_Alignment", "Time_Closeness"]].values, dtype=torch.float32)
    labels = torch.tensor(np.expand_dims(df["Linked"].values, axis=-1), dtype=torch.float32)

    return torch.utils.data.TensorDataset(features, labels)


def train_model(dataset, model):
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    running_loss = 0.0
    last_loss = 0.0

    for i in range(len(dataset)):
        X_sample, y_sample = dataset[i]

        optimizer.zero_grad()

        y_pred = model(X_sample)
        # print(f"y_pred={y_pred}, y_sample={y_sample}")

        loss = loss_fn(y_pred, y_sample)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 0:
            print("Iteration: {}, Loss: {} ".format(i+1, last_loss))
    
    return model


if __name__ == "__main__":
    df = load_dataset()
    # analyze_data(df)
    model = ALSModel()
    dataset = prepare_features(df)
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    print(len(dataset), len(train_dataset))
    train_model(train_dataset, validation_dataset, model)

