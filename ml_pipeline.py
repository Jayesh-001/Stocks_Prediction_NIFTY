# ml_pipeline.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import copy
from models import StockLSTM

def train_and_predict(df, features, lookback_window, epochs, patience=15, st_progress_bar=None, st_status_text=None):
    """Handles ML pipeline: Train/Val split, Early Stopping, and Directional Accuracy."""
    
    data_values = df[features].values
    
    # 1. Scaling
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = feature_scaler.fit_transform(data_values)
    
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(df[['Close']]) 

    # 2. Sequence Generation
    X, y = [], []
    for i in range(lookback_window, len(scaled_data)):
        X.append(scaled_data[i-lookback_window:i, :])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # 3. Chronological Train / Validation Split (85% / 15%)
    split_idx = int(len(X_tensor) * 0.85)
    X_train, y_train = X_tensor[:split_idx], y_tensor[:split_idx]
    X_val, y_val = X_tensor[split_idx:], y_tensor[split_idx:]

    # 4. Initialize Model & Early Stopping Variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockLSTM(input_size=len(features), hidden_size=64, num_layers=2, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    stopped_epoch = epochs

    # 5. Training Loop with Validation
    for epoch in range(epochs):
        # -- Training Phase --
        model.train()
        X_train_dev, y_train_dev = X_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        out_train = model(X_train_dev)
        loss_train = criterion(out_train, y_train_dev)
        loss_train.backward()
        optimizer.step()
        
        # -- Validation Phase --
        model.eval()
        with torch.no_grad():
            X_val_dev, y_val_dev = X_val.to(device), y_val.to(device)
            out_val = model(X_val_dev)
            loss_val = criterion(out_val, y_val_dev)
            
        # -- Early Stopping Logic --
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
            stopped_epoch = epoch + 1
        else:
            patience_counter += 1
            
        # UI Updates
        if st_progress_bar and st_status_text:
            progress = min((epoch + 1) / epochs, 1.0)
            st_progress_bar.progress(progress)
            st_status_text.text(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss_train.item():.6f} | Val Loss: {loss_val.item():.6f}")

        if patience_counter >= patience:
            if st_status_text:
                st_status_text.text(f"Early Stopping triggered at Epoch {epoch+1}! Restoring best weights from Epoch {stopped_epoch}.")
            break

    # 6. Load Best Weights & Calculate Directional Accuracy
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    model.eval()
    with torch.no_grad():
        preds_val = model(X_val.to(device)).cpu()
        
    # Logic: Did it correctly predict the UP or DOWN movement compared to the PREVIOUS day?
    # To compare validation day 'i', we need actual price from day 'i-1'
    prev_actuals = torch.cat([y_train[-1:], y_val[:-1]]) 
    
    actual_moves = y_val - prev_actuals
    predicted_moves = preds_val - prev_actuals
    
    # If actual move and predicted move have the same sign (both positive or both negative), multiplying them yields a positive number.
    correct_directions = ((actual_moves * predicted_moves) > 0).sum().item()
    directional_accuracy = (correct_directions / len(y_val)) * 100

    # 7. Inference (Predict Tomorrow)
    last_sequence = scaled_data[-lookback_window:]
    last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_scaled = model(last_sequence_tensor)
        predicted_price = target_scaler.inverse_transform(predicted_scaled.cpu().numpy())[0][0]

    current_price = df['Close'].iloc[-1].item()
    
    return current_price, predicted_price, stopped_epoch, directional_accuracy