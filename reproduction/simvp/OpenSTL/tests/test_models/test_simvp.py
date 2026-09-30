import pytest
import torch

from openstl.models import SimVP_Model


def test_invalid_model_type():
    """An unsupported model_type must raise AssertionError inside MetaBlock."""
    with pytest.raises(AssertionError):
        SimVP_Model(in_shape=(4, 1, 16, 16), model_type='unknown')


@pytest.mark.parametrize('model_type', ['gSTA', 'IncepU', 'ConvNeXt', 'Swin'])
def test_forward_shape(model_type):
    """SimVP_Model must preserve (B, T, C, H, W) through encode-translate-decode.

    H, W are downsampled by 2**(N_S/2); with N_S=4 the spatial size must be
    divisible by 4, so 16x16 is used here.
    """
    in_shape = (4, 1, 16, 16)
    model = SimVP_Model(
        in_shape=in_shape, model_type=model_type,
        hid_S=4, hid_T=8, N_S=4, N_T=2,
    )
    model.eval()
    x = torch.randn(2, *in_shape)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 4, 1, 16, 16)


def test_output_frames_equal_input_frames():
    """The model always predicts T frames equal to the input T; longer rollout
    (aft > pre) is handled by the SimVP method, not the model."""
    in_shape = (4, 1, 16, 16)
    model = SimVP_Model(
        in_shape=in_shape, model_type='gSTA',
        hid_S=4, hid_T=8, N_S=4, N_T=2,
    )
    model.eval()
    x = torch.randn(2, 4, 1, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
