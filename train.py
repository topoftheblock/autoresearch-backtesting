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
            nn.BatchNorm1d(64),  # Added Batch Normalization
            nn.Dropout(0.2),  # Regularization
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),  # Added Batch Normalization
            nn.Dropout(0.2),  # Regularization
            nn.Linear(32, 1)
            # Removed Sigmoid for BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x)

def train():
    # Load data
    train_df = pd.read_csv('data/train.csv', index_col=0)
    features = ['Returns', 'Vol_20', 'SMA_10', 'SMA_50']
    
    X_train = torch.tensor(train_df[features].values, dtype=torch.float32)
    y_train = torch.tensor(train_df['Target'].values, dtype=torch.float32).unsqueeze(1)
    
    model = FinanceModel(input_dim=len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Compute class weights to handle imbalance
    class_counts = train_df['Target'].value_counts()
    pos_weight = class_counts[0] / class_counts[1]
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))

    print("Starting training...")
    for epoch in range(500):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            
    torch.save(model.state_dict(), 'model_weights.pt')
    print("Training complete. Model saved.")

if __name__ == '__main__':
    train()