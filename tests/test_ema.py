import pytest

import torch

from torch_ema import ExponentialMovingAverage


@pytest.mark.parametrize("decay", [0.995, 0.9])
@pytest.mark.parametrize("use_num_updates", [True, False])
@pytest.mark.parametrize("explicit_params", [True, False])
def test_val_error(decay, use_num_updates, explicit_params):
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
        if explicit_params:
            ema.update(model.parameters())
        else:
            ema.update()

    # Validation: original
    model.eval()
    logits = model(x_val)
    loss_orig = torch.nn.functional.cross_entropy(logits, y_val)
    print(f"Original loss: {loss_orig}")

    # Validation: with EMA
    # First save original parameters before replacing with EMA version
    if explicit_params:
        ema.store(model.parameters())
    else:
        ema.store()
    # Copy EMA parameters to model
    if explicit_params:
        ema.copy_to(model.parameters())
    else:
        ema.copy_to()
    logits = model(x_val)
    loss_ema = torch.nn.functional.cross_entropy(logits, y_val)

    print(f"EMA loss: {loss_ema}")
    assert loss_ema < loss_orig, "EMA loss wasn't lower"

    # Test restore
    if explicit_params:
        ema.restore(model.parameters())
    else:
        ema.restore()
    model.eval()
    logits = model(x_val)
    loss_orig2 = torch.nn.functional.cross_entropy(logits, y_val)
    assert torch.allclose(loss_orig, loss_orig2), \
        "Restored model wasn't the same as stored model"


@pytest.mark.parametrize("explicit_params", [True, False])
def test_contextmanager(explicit_params):
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
        decay=0.99,
    )

    # Train for a few epochs
    model.train()
    for _ in range(20):
        logits = model(x_train)
        loss = torch.nn.functional.cross_entropy(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if explicit_params:
            ema.update(model.parameters())
        else:
            ema.update()

    final_weight = model.weight.clone().detach()

    # Validation: original
    model.eval()
    logits = model(x_val)
    loss_orig = torch.nn.functional.cross_entropy(logits, y_val)
    print(f"Original loss: {loss_orig}")

    # Validation: with EMA
    if explicit_params:
        cm = ema.average_parameters(model.parameters())
    else:
        cm = ema.average_parameters()

    with cm:
        logits = model(x_val)
        loss_ema = torch.nn.functional.cross_entropy(logits, y_val)

    print(f"EMA loss: {loss_ema}")
    assert loss_ema < loss_orig, "EMA loss wasn't lower"
    assert torch.all(model.weight == final_weight), "Restore failed"


@pytest.mark.parametrize("decay", [0.995, 0.9, 0.0, 1.0])
@pytest.mark.parametrize("use_num_updates", [True, False])
@pytest.mark.parametrize("explicit_params", [True, False])
def test_store_restore(decay, use_num_updates, explicit_params):
    model = torch.nn.Linear(10, 2)
    ema = ExponentialMovingAverage(
        model.parameters(),
        decay=decay,
        use_num_updates=use_num_updates
    )
    orig_weight = model.weight.clone().detach()
    if explicit_params:
        ema.store(model.parameters())
    else:
        ema.store()
    with torch.no_grad():
        model.weight.uniform_(0.0, 1.0)
    if explicit_params:
        ema.restore(model.parameters())
    else:
        ema.restore()
    assert torch.all(model.weight == orig_weight)


@pytest.mark.parametrize("decay", [0.995, 0.9, 0.0, 1.0])
@pytest.mark.parametrize("explicit_params", [True, False])
def test_update(decay, explicit_params):
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
    if explicit_params:
        ema.update(model.parameters())
    else:
        ema.update()
    assert torch.all(model.weight == 1.0), "ema.update changed model weights"
    if explicit_params:
        ema.copy_to(model.parameters())
    else:
        ema.copy_to()
    assert torch.allclose(
        model.weight,
        torch.full(size=(1,), fill_value=(1.0 - decay))
    ), "average was wrong"


def test_explicit_params():
    model = torch.nn.Linear(10, 2)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
    model2 = torch.nn.Linear(10, 2)
    with torch.no_grad():
        model2.weight.fill_(1.0)
    ema.update(model2.parameters())
    ema.copy_to()
    assert not torch.all(model.weight == 0.0)


def test_some_untrainable():
    class Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x = torch.nn.Parameter(torch.randn(3))
            self.y = torch.nn.Parameter(torch.randn(3))
            self.y.requires_grad_(False)

        def forward(self, x):
            return self.x * x + self.y

    model = Mod()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
    ema.update()
    with torch.no_grad():
        model.x *= 1.1
    ema.update()
    ema.store()
    ema.copy_to()


def test_to():
    m = torch.nn.Linear(11, 3)
    ema = ExponentialMovingAverage(m.parameters(), decay=0.9)
    assert ema.shadow_params[0].dtype == torch.get_default_dtype()
    ema.to(dtype=torch.float16)
    assert ema.shadow_params[0].dtype == torch.float16
    ema.store()
    # we store whatever we get
    assert ema.collected_params[0].dtype == torch.get_default_dtype()
    m = m.to(torch.float16)
    ema.store(m.parameters())
    assert ema.collected_params[0].dtype == torch.float16
    ema.to(dtype=torch.float64)
    assert ema.collected_params[0].dtype == torch.float64
    assert ema.shadow_params[0].dtype == torch.float64
