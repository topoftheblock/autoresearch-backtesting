import torch
import torch.nn as nn
import pandas as pd

class FinanceModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),  # Normalize inputs
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),  # Batch Normalization
            nn.Dropout(0.2),     # Regularization
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),  # Batch Normalization
            nn.Dropout(0.2),     # Regularization
            nn.Linear(32, 1)
            # Removed Sigmoid for BCEWithLogitsLoss compatibility
        )

    def forward(self, x):
        return self.net(x)

class ProfitAwareLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        # reduction='none' is CRITICAL here. It stops PyTorch from immediately averaging
        # the loss so we can multiply each day's loss by that day's specific return weight.
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def forward(self, logits, targets, future_returns):
        # 1. Calculate standard BCE loss per sample
        base_loss = self.bce(logits, targets)
        
        # 2. Extract the magnitude (absolute value) of the next day's return
        weights = torch.abs(future_returns)
        
        # 3. Normalize weights so their average is 1.0.
        # This prevents the overall loss scale from crashing (since returns are tiny numbers like 0.01),
        # which would otherwise break our optimizer's learning rate.
        weights = weights / (weights.mean() + 1e-8)
        
        # 4. Apply the weights to the loss and take the mean
        weighted_loss = base_loss * weights
        return weighted_loss.mean()

def train():
    # Load data
    train_df = pd.read_csv('data/train.csv', index_col=0)
    
    # Calculate the continuous future return (the magnitude of the target)
    train_df['Future_Return'] = train_df['Returns'].shift(-1)
    
    # Drop the final row because its 'Future_Return' is NaN
    train_df = train_df.dropna()
    
    features = ['Returns', 'Vol_20', 'SMA_10', 'SMA_50']
    
    # Convert inputs and targets to PyTorch Tensors
    X_train = torch.tensor(train_df[features].values, dtype=torch.float32)
    y_train = torch.tensor(train_df['Target'].values, dtype=torch.float32).unsqueeze(1)
    future_returns = torch.tensor(train_df['Future_Return'].values, dtype=torch.float32).unsqueeze(1)
    
    model = FinanceModel(input_dim=len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Compute class weights to handle Long/Short imbalance
    class_counts = train_df['Target'].value_counts()
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32)
    
    # Initialize our new custom loss
    criterion = ProfitAwareLoss(pos_weight=pos_weight)

    print("Starting training...")
    for epoch in range(500):
        model.train()  # Set to train mode
        optimizer.zero_grad()
        
        outputs = model(X_train)
        
        # Pass the future returns into the loss function alongside outputs and targets
        loss = criterion(outputs, y_train, future_returns)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/500, Loss: {loss.item():.4f}")

    # Save weights
    torch.save(model.state_dict(), 'model_weights.pt')
    print("Training complete. Weights saved to model_weights.pt")

if __name__ == '__main__':
    train()