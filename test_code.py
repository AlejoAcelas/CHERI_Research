import pytest
import torch
from model_setup import MNIST_CNN
from copy import deepcopy

torch.set_grad_enabled(False)

@pytest.fixture
def small_model():
    model = MNIST_CNN(num_classes=2, channels1=4, channels2=8)
    model.eval()
    return model

def test_run_from_to(small_model: MNIST_CNN):
    model = small_model
    input = torch.randn(64, 1, 28, 28)
    
    out = model(input)
    for start in model.hook_list[:-1]:
        mid_acts = model.run_from_to(input, start='input', end=start)
        out2 = model.run_from_to(mid_acts, start=start, end='fc2')
        torch.testing.assert_close(out, out2)

def test_run_with_weights(small_model: MNIST_CNN):
    model = small_model
    input = torch.randn(64, 1, 28, 28)

    # Conv1 tests
    filter_weights = model.conv1.weight.data[0:2]
    filter_bias = model.conv1.bias.data[0:2]

    for end in model.hook_list[1:]:
        print('Examining =', end)
        acts_out = model.run_from_to(input, end=end)
        acts_out2 = model.run_with_new_weights(input, layer='conv1', idx=[0, 1], weight=filter_weights, bias=filter_bias, end=end)
        torch.testing.assert_close(acts_out, acts_out2)


    model_copy = deepcopy(model)
    model_copy.eval()
    new_filter_weights = torch.randn_like(filter_weights)
    new_filter_bias = torch.randn_like(filter_bias)
    model_copy.conv1.weight.data[0:2] = new_filter_weights
    model_copy.conv1.bias.data[0:2] = new_filter_bias

    torch.testing.assert_close(model_copy.conv1.weight.data[0:2], new_filter_weights)
    torch.testing.assert_close(model_copy.conv1.bias.data[0:2], new_filter_bias)

    for end in model.hook_list[1:]:
        print('Examining =', end)
        acts_out = model_copy.run_from_to(input, end=end)
        acts_out2 = model.run_with_new_weights(input, layer='conv1', idx=[0, 1], weight=new_filter_weights, bias=new_filter_bias, end=end)
        torch.testing.assert_close(acts_out, acts_out2)
