import torch
import pandas as pd
import numpy as np

from training.model import ALSModel

CSV_FILE_PATH = "./data/03_CORE_training_dataset_FINAL.csv"

NUM_EPOCHS = 30
LEARNING_RATE = 0.0003
EARLY_STOPPING_THRESHOLD = 5
FEATURE_FIELDS = ["Semantic_Similarity", "Emotional_Alignment", "Time_Closeness"]
LABEL_FIELD = "Linked"
MODEL_SAVE_PATH = "data/models/als.pt"


def load_dataset():
    return pd.read_csv(CSV_FILE_PATH)

def analyze_data(df):
    print(df.head())
    print(df.describe())
    print(f"Number of causal edges: {len(df[df['Linked'] == 1])}")
    print(f"Number of non-causal edges: {len(df[df['Linked'] == 0])}")

def create_model():
    return ALSModel()

def create_dataset(df, story_id=None):
    if story_id:
        story_df = df[df["story_id"] == story_id]
        df = story_df

    features_df = df[FEATURE_FIELDS]
    labels_df = df[LABEL_FIELD]

    print(features_df)

    features = torch.tensor(features_df.values, dtype=torch.float32)
    labels = torch.tensor(np.expand_dims(labels_df.values, axis=-1), dtype=torch.float32)

    return torch.utils.data.TensorDataset(features, labels)


def train_model(train_dataset, validation_dataset, model):
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, threshold=0.0001)

    running_loss = 0.0
    last_loss = 0.0

    best_validation_loss = float("inf")
    epochs_with_no_improvement = 0
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}")
        model.train(True)
        for i in range(len(train_dataset)):
            X_sample, y_sample = train_dataset[i]

            optimizer.zero_grad()

            y_pred = model(X_sample)

            loss = loss_fn(y_pred, y_sample)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i != 0 and i % 500 == 0:
                last_loss = running_loss / 500
                running_loss = 0.0
                print("Training: {}, Loss: {} ".format((i // 500) + 1, last_loss))
        
        model.eval()

        # Against Validation Set
        with torch.no_grad():
            for i in range(len(validation_dataset)):
                X_sample, y_sample = validation_dataset[i]
                y_pred = model(X_sample)
                loss = loss_fn(y_pred, y_sample)

                running_loss += loss.item()
            
            last_loss = running_loss / len(validation_dataset)
            running_loss = 0.0
            print("Validation: {}, Loss: {} ".format(i+1, last_loss))
        
        if last_loss < best_validation_loss:
            best_validation_loss = last_loss
            epochs_with_no_improvement = 0
            torch.save(model, MODEL_SAVE_PATH)
            print(f"Model saved to {MODEL_SAVE_PATH}")
        else:
            epochs_with_no_improvement += 1
        
        if epochs_with_no_improvement == EARLY_STOPPING_THRESHOLD:
            print("Early stopping activated")
            break

        lr_scheduler.step(metrics=last_loss)
    
    return model


if __name__ == "__main__":
    df = load_dataset()
    analyze_data(df)
    model = ALSModel(ignore_feature=None)
    dataset = create_dataset(df, story_id=None)
    print(dataset[0])
    # train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    # print(f"Training Dataset: {len(train_dataset)}, Validation Dataset: {len(validation_dataset)}")
    # train_model(train_dataset, validation_dataset, model)

