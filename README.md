# pytorch_ema

A very small library for computing exponential moving averages of model
parameters.

This library was written for personal use. Nevertheless, if you run into issues
or have suggestions for improvement, feel free to open either a new issue or
pull request.

## Installation

```
pip install -U git+https://github.com/fadel/pytorch_ema
```

## Example

```python
import torch
import torch.nn.functional as F

from torch_ema import ExponentialMovingAverage

torch.manual_seed(0)
x_train = torch.rand((100, 10))
y_train = torch.rand(100).round().long()
x_val = torch.rand((100, 10))
y_val = torch.rand(100).round().long()
model = torch.nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

# Train for a few epochs
model.train()
for _ in range(20):
    logits = model(x_train)
    loss = F.cross_entropy(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ema.update(model.parameters())

# Validation: original
model.eval()
logits = model(x_val)
loss = F.cross_entropy(logits, y_val)
print(loss.item())

# Validation: with EMA
# First save original parameters before replacing with EMA version
ema.store(model.parameters())
# Copy EMA parameters to model
ema.copy_to(model.parameters())
logits = model(x_val)
loss = F.cross_entropy(logits, y_val)
print(loss.item())
# Restore original parameters to resume training later
ema.restore(model.parameters())
```
