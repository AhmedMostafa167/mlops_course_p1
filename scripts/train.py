"""
Trains a tiny regression model (ride duration from distance + passengers)
and saves it in two forms so you can load it in your API without
needing this training code at all.

Run: python train.py
Produces: model.pt (state_dict)
"""

import torch
import torch.nn as nn
from pathlib import Path
from src.model import RideDurationNet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model_full.pt"

torch.manual_seed(0)
n = 2000
distance_km = torch.rand(n, 1) * 20          # 0-20 km
passengers = torch.randint(1, 5, (n, 1)).float()
X = torch.cat([distance_km, passengers], dim=1)
y = distance_km * 2.5 + passengers * 0.5 + torch.randn(n, 1) * 0.5  # fake target


model = RideDurationNet()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

for epoch in range(200):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

print(f"final loss: {loss.item():.4f}")

model.eval()

torch.save(model.state_dict(), MODEL_PATH)
print("Saved model.pt (state_dict)")