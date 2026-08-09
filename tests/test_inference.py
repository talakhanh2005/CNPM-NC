import torch

from openemotion.model import create_model


def test_model_output_shape():
    model = create_model(7)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert output.shape == (2, 7)
