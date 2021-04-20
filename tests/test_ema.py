import pytest

import torch

from torch_ema import ExponentialMovingAverage


@pytest.mark.parametrize("decay", [0.995, 0.9])
@pytest.mark.parametrize("use_num_updates", [True, False])
def test_val_error(decay, use_num_updates):
    """Confirm that EMA validation error is lower than raw validation error."""
    torch.manual_seed(0)
    x_train = torch.rand((100, 10))
    y_train = torch.rand(100).round().long()
    x_val = torch.rand((100, 10))
    y_val = torch.rand(100).round().long()
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    ema = ExponentialMovingAverage(
        model.parameters(),
        decay=decay,
        use_num_updates=use_num_updates
    )

    # Train for a few epochs
    model.train()
    for _ in range(20):
        logits = model(x_train)
        loss = torch.nn.functional.cross_entropy(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model.parameters())

    # Validation: original
    model.eval()
    logits = model(x_val)
    loss_orig = torch.nn.functional.cross_entropy(logits, y_val)
    print(f"Original loss: {loss_orig}")

    # Validation: with EMA
    # First save original parameters before replacing with EMA version
    ema.store(model.parameters())
    # Copy EMA parameters to model
    ema.copy_to(model.parameters())
    logits = model(x_val)
    loss_ema = torch.nn.functional.cross_entropy(logits, y_val)

    print(f"EMA loss: {loss_ema}")
    assert loss_ema < loss_orig, "EMA loss wasn't lower"

    # Test restore
    ema.restore(model.parameters())
    model.eval()
    logits = model(x_val)
    loss_orig2 = torch.nn.functional.cross_entropy(logits, y_val)
    assert torch.allclose(loss_orig, loss_orig2), \
        "Restored model wasn't the same as stored model"


@pytest.mark.parametrize("decay", [0.995, 0.9, 0.0, 1.0])
@pytest.mark.parametrize("use_num_updates", [True, False])
def test_store_restore(decay, use_num_updates):
    model = torch.nn.Linear(10, 2)
    ema = ExponentialMovingAverage(
        model.parameters(),
        decay=decay,
        use_num_updates=use_num_updates
    )
    orig_weight = model.weight.clone().detach()
    ema.store(model.parameters())
    with torch.no_grad():
        model.weight.uniform_(0.0, 1.0)
    ema.restore(model.parameters())
    assert torch.all(model.weight == orig_weight)


@pytest.mark.parametrize("decay", [0.995, 0.9, 0.0, 1.0])
def test_update(decay):
    model = torch.nn.Linear(10, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = ExponentialMovingAverage(
        model.parameters(),
        decay=decay,
        use_num_updates=False
    )
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model.parameters())
    assert torch.all(model.weight == 1.0), "ema.update changed model weights"
    ema.copy_to(model.parameters())
    assert torch.allclose(
        model.weight,
        torch.full(size=(1,), fill_value=(1.0 - decay))
    ), "average was wrong"
