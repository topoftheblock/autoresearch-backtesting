import torch
import torch.nn as nn
import pandas as pd

class FinanceModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, 64, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
            # Removed Sigmoid for BCEWithLogitsLoss compatibility
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Reshape for GRU: (batch_size, 1, features)
        _, h_n = self.gru(x)
        return self.net(h_n.squeeze(0))

class ProfitAwareLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def forward(self, logits, targets, future_returns):
        base_loss = self.bce(logits, targets)
        weights = torch.abs(future_returns)
        weights = weights / (weights.mean() + 1e-8)
        weighted_loss = base_loss * weights
        return weighted_loss.mean()

def train():
    # Load data
    train_df = pd.read_csv('data/train.csv', index_col=0)
    val_df = pd.read_csv('data/val.csv', index_col=0) # <--- Added this
    
    # Calculate the continuous future return (the magnitude of the target)
    train_df['Future_Return'] = train_df['Returns'].shift(-1)
    train_df = train_df.dropna()

    val_df['Future_Return'] = val_df['Returns'].shift(-1) # <--- Added this
    val_df = val_df.dropna() # <--- Added this
    
    features = ['Returns', 'Vol_20', 'SMA_10', 'SMA_50']
    
    X_train = torch.tensor(train_df[features].values, dtype=torch.float32)
    y_train = torch.tensor(train_df['Target'].values, dtype=torch.float32).unsqueeze(1)
    future_returns = torch.tensor(train_df['Future_Return'].values, dtype=torch.float32).unsqueeze(1)
    
    model = FinanceModel(input_dim=len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    class_counts = train_df['Target'].value_counts()
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32)
    
    criterion = ProfitAwareLoss(pos_weight=pos_weight)

    print("Starting training...")
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train, future_returns)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/1000, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'model_weights.pt')
    print("Training complete. Weights saved to model_weights.pt")

if __name__ == '__main__':
    train()