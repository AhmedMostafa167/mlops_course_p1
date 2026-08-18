"""
Trains a tiny regression model (ride duration from distance + passengers)
and saves it in two forms so you can load it in your API without
needing this training code at all.

Run: python train.py
Produces: model.pt (state_dict) and model_full.pt (full model, easiest to load)
"""

import torch
import torch.nn as nn

# ---- 1. Fake data ----------------------
torch.manual_seed(0)
n = 2000
distance_km = torch.rand(n, 1) * 20          # 0-20 km
passengers = torch.randint(1, 5, (n, 1)).float()
X = torch.cat([distance_km, passengers], dim=1)
y = distance_km * 2.5 + passengers * 0.5 + torch.randn(n, 1) * 0.5  # fake target

# ---- 2. Model --------------------------------------------------------------
class RideDurationNet(nn.Module):
    def __init__(self, in_features: int = 2, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = RideDurationNet()

# ---- 3. Train ---------------------------------------------------------------
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

# ---- 4. Save for deployment --------------------------------------------------
model.eval()

# Option A: state_dict only (smaller, but the loading code needs the class def)
torch.save(model.state_dict(), "models/model.pt")

# Option B: full model object (loads with zero code duplication — recommended
# for your mini project since it keeps src/api.py simple)
torch.save(model, "models/model_full.pt")

print("Saved model.pt (state_dict) and model_full.pt (full model)")