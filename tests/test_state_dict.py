import pytest

import copy

import torch

from torch_ema import ExponentialMovingAverage


@pytest.mark.parametrize("decay", [0.995])
@pytest.mark.parametrize("use_num_updates", [True, False])
@pytest.mark.parametrize("explicit_params", [True, False])
def test_state_dict(decay, use_num_updates, explicit_params):
    model = torch.nn.Linear(10, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = ExponentialMovingAverage(
        model.parameters(),
        decay=decay,
        use_num_updates=False
    )
    state_dict = copy.deepcopy(ema.state_dict())

    model2 = torch.nn.Linear(10, 2, bias=False)
    ema2 = ExponentialMovingAverage(model2.parameters(), decay=0.0)
    ema2.load_state_dict(state_dict)
    assert ema2.decay == decay
    assert torch.allclose(ema2.shadow_params[0], ema.shadow_params[0])

    with torch.no_grad():
        model2.weight.fill_(1.0)
    if explicit_params:
        ema2.update(model2.parameters())
    else:
        ema2.update()
    assert torch.all(model2.weight == 1.0), "ema.update changed model weights"

    ema.load_state_dict(ema2.state_dict())

    if explicit_params:
        ema.copy_to(model.parameters())
    else:
        ema.copy_to()
    assert torch.allclose(
        model.weight,
        torch.full(size=(1,), fill_value=(1.0 - decay))
    ), "average was wrong"


def test_state_dict_types():
    m1 = torch.nn.Linear(10, 2, bias=False)
    m2 = torch.nn.Linear(10, 2, bias=False)
    m2.to(torch.float16)
    ema1 = ExponentialMovingAverage(m1.parameters(), decay=0.9)
    ema2 = ExponentialMovingAverage(m2.parameters(), decay=0.9)
    ema1.update()
    ema2.update()
    ema2.load_state_dict(ema1.state_dict())
    ema1.copy_to()
    ema2.copy_to()
    assert m1.weight.dtype == torch.get_default_dtype()
    assert m2.weight.dtype == torch.float16
    assert torch.allclose(m1.weight.to(torch.float16), m2.weight)


def test_bad_state_dict1():
    m = torch.nn.Linear(10, 2, bias=False)
    ema = ExponentialMovingAverage(m.parameters(), decay=0.9)
    sd = ema.state_dict()
    sd["shadow_params"][0] = torch.zeros(3, 7)
    # it doesn't raise at loading, since it can't know shapes.
    ema.load_state_dict(sd)
    with pytest.raises(RuntimeError):
        ema.copy_to()
    # make sure it didn't change
    assert torch.any(m.weight.abs() > 0)


def test_bad_state_dict2():
    m = torch.nn.Linear(10, 2, bias=False)
    ema = ExponentialMovingAverage(m.parameters(), decay=0.9)
    sd = ema.state_dict()
    sd["shadow_params"] = sd["shadow_params"][:-1]
    with pytest.raises(ValueError):
        ema.load_state_dict(sd)
