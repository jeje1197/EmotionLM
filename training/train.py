import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from training.model import ALSModel

# UPDATED: Using corrected dataset with days-based temporal normalization
CSV_FILE_PATH = "./data/03_CORE_training_dataset_CORRECTED.csv"

# Hyperparameters - Experiment with these!
NUM_EPOCHS = 100  # Increased for longer training
LEARNING_RATE = 0.001  # Can experiment: [0.0001, 0.0003, 0.001, 0.003, 0.01]
EARLY_STOPPING_THRESHOLD = 15  # Increased patience for longer training

FEATURE_FIELDS = ["Semantic_Similarity", "Emotional_Alignment", "Time_Closeness"]
LABEL_FIELD = "Linked"
MODEL_SAVE_PATH = "data/models/als.pt"
PLOT_SAVE_PATH = "research/images/loss_curve.png"

SAVE_MODEL = True  # Changed to True to save best model
SAVE_PLOT = True  # Changed to True to save training curves

def load_dataset():
    return pd.read_csv(CSV_FILE_PATH)

def analyze_data(df):
    print(df.head())
    print(df.describe())
    print(f"Number of causal edges: {len(df[df['Linked'] == 1])}")
    print(f"Number of non-causal edges: {len(df[df['Linked'] == 0])}")

def create_dataset(df, story_id=None):
    if story_id:
        df = df[df["story_id"] == story_id]

    features_df = df[FEATURE_FIELDS]
    labels_df = df[LABEL_FIELD]

    features = torch.tensor(features_df.values, dtype=torch.float32)
    labels = torch.tensor(np.expand_dims(labels_df.values, axis=-1), dtype=torch.float32)

    return torch.utils.data.TensorDataset(features, labels)

def train_model(train_dataset, validation_dataset, model):
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, threshold=0.0001)

    history = {
        "train_loss": [],
        "val_loss": []
    }

    best_validation_loss = float("inf")
    epochs_with_no_improvement = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Training
        model.train()
        train_running_loss = 0.0
        for i in range(len(train_dataset)):
            X_sample, y_sample = train_dataset[i]
            optimizer.zero_grad()
            y_pred = model(X_sample)
            loss = loss_fn(y_pred, y_sample)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
        
        avg_train_loss = train_running_loss / len(train_dataset)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for i in range(len(validation_dataset)):
                X_sample, y_sample = validation_dataset[i]
                y_pred = model(X_sample)
                v_loss = loss_fn(y_pred, y_sample)
                val_running_loss += v_loss.item()
        
        avg_val_loss = val_running_loss / len(validation_dataset)
        history["val_loss"].append(avg_val_loss)
        
        print(f"Loss - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_validation_loss:
            best_validation_loss = avg_val_loss
            epochs_with_no_improvement = 0
            if SAVE_MODEL:
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"New Best Model Saved!")
        else:
            epochs_with_no_improvement += 1
        
        if epochs_with_no_improvement == EARLY_STOPPING_THRESHOLD:
            print("Early stopping activated")
            break

        lr_scheduler.step(avg_val_loss)
    
    return model, history

def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', color='#1f77b4', lw=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='#ff7f0e', lw=2)
    plt.title('ALS Training Convergence', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('BCE Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    if SAVE_PLOT:
        # Create the directory if it doesn't exist
        if not os.path.exists(os.path.dirname(PLOT_SAVE_PATH)):
            os.makedirs(os.path.dirname(PLOT_SAVE_PATH))

        plt.savefig(PLOT_SAVE_PATH)
        print(f"Plot saved to {PLOT_SAVE_PATH}")
    plt.show()

def evaluate_baseline(dataset, model):
    """
    Evaluates the model's performance before any training occurs.
    """
    model.eval()
    loss_fn = torch.nn.BCELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            X, y = dataset[i]
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            
    avg_baseline_loss = total_loss / len(dataset)
    return avg_baseline_loss


if __name__ == "__main__":
    # Load Data
    df = load_dataset()
    analyze_data(df)
    
    model = ALSModel(ignore_feature=None)
    dataset = create_dataset(df, story_id=None)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Training Dataset: {len(train_data)}, Validation Dataset: {len(val_data)}")
    
    # Capture Baseline Results
    baseline_val_loss = evaluate_baseline(val_data, model)
    print(f"\nPre-training Validation Loss: {baseline_val_loss:.4f}")
    
    # Training
    trained_model, training_history = train_model(train_data, val_data, model)
    
    # Add Baseline into History for Plot
    training_history["train_loss"].insert(0, baseline_val_loss)
    training_history["val_loss"].insert(0, baseline_val_loss)
    
    # Visualization
    plot_history(training_history)
    
    print(f"Baseline Loss: {baseline_val_loss:.4f}")
    print(f"Final Best Val Loss: {min(training_history['val_loss']):.4f}")

    weights = trained_model.slp.weight[0].detach().tolist()
    bias = trained_model.slp.bias.item()

    print(f"Weight SS (Semantic): {weights[0]:.4f}")
    print(f"Weight SE (Emotion):  {weights[1]:.4f}")
    print(f"Weight ST (Time):     {weights[2]:.4f}")
    print(f"Bias:                 {bias:.4f}")