import torch
from torch import Tensor, optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from collections import defaultdict
import numpy as np
import einops
from collections.abc import Iterable
from transformer_lens import HookedTransformerConfig
from typing import List, Tuple, Dict, Optional, Literal, Union, Callable
from jaxtyping import Float, Int, Bool
from tqdm.notebook import tqdm
from PIL import Image, ImageDraw


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

### MODEL SETUP ###

class MNIST_CNN(nn.Module):  
    
    hook_list = ['input', 'conv1', 'drop1', 'pool1', 'conv2', 'drop2', 'pool2', 'fc1', 'drop3', 'fc2']
    hook_list_type = Literal['input', 'conv1', 'drop1', 'pool1', 'conv2', 'drop2', 'pool2', 'fc1', 'drop3', 'fc2']

    def __init__(self, num_classes: int, channels1=32, channels2=16, dropout=0.2):
        super(MNIST_CNN, self).__init__()

        self.n_features = channels2*5*5
        self.conv1 = nn.Conv2d(1, channels1, 3, 1)
        self.conv2 = nn.Conv2d(channels1, channels2, 3, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2) # Redundant, but useful for hooking
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(self.n_features, 200)
        self.drop3 = nn.Dropout(p=dropout/2)
        self.fc2 = nn.Linear(200, num_classes)

    def forward(self, X):
        X = self.pool1(self.drop1(F.relu(self.conv1(X))))
        X = self.pool2(self.drop2(F.relu(self.conv2(X))))
        X = X.view(-1, self.n_features)
        X = self.drop3(F.relu(self.fc1(X)))
        X = self.fc2(X)
        return X
    
    def run_from_to(self,
                    X: Float[Tensor, 'batch ...'],
                    start: hook_list_type = 'input',
                    end: hook_list_type = 'fc2',
                    skip_relu_end: bool = False,
        ):
        start_idx = self.hook_list.index(start)
        end_idx = self.hook_list.index(end)
        assert start_idx <= end_idx, f"start module must come before end module. {start} -> {end}"
        
        for mod in self.hook_list[start_idx+1:end_idx+1]:
            module = self.__getattr__(mod)
            X = module(X)
            if mod in ['conv1', 'conv2', 'fc1']:
                if mod == end and skip_relu_end:
                    pass
                else:
                    X = F.relu(X)
            if mod == 'pool2':
                X = X.view(-1, self.n_features)
        return X

    def run_with_new_weights(self,
                         X: Float[Tensor, 'batch channels height width'],
                         layer: Literal['conv1', 'conv2', 'fc1', 'fc2'],
                         idx: Union[int, List[int]],
                         weight: Float[Tensor, '...'], 
                         bias: Float[Tensor, '...'] = None,
                         end: hook_list_type = 'fc2',
        ):
        if type(idx) == int:
            idx = [idx] # Makes sure shapes are correct

        layer_idx = self.hook_list.index(layer)
        prev_layer = self.hook_list[layer_idx-1]
        layer_in = self.run_from_to(X, end=prev_layer)
        layer_out = self.run_from_to(layer_in, start=prev_layer, end=layer)

        if 'conv' in layer:
            new_out = F.conv2d(input=layer_in, weight=weight, bias=bias)
        elif 'fc' in layer:
            new_out = F.linear(layer_in, weight, bias)
        else:
            raise ValueError(f"Layer {layer} not supported")
        
        layer_out[:, idx] = new_out if layer == 'fc2' else F.relu(new_out)
        return self.run_from_to(layer_out, start=layer, end=end)
    

### DATASETS ###

def create_blank_image(width=28, height=28, color=0):
    return Image.new("1", (width, height), color)

def draw_polygon_polar(image, vertices: int, R: int, r: int, color=1, 
                          offset_angle=np.pi/4, offset_x=0, offset_y=0):
    draw = ImageDraw.Draw(image)
    radii = np.random.rand(vertices) * (R - r) + r
    angles = np.linspace(0, 2 * np.pi, vertices, endpoint=False) + offset_angle
    points = np.stack([radii * np.cos(angles) + 14 + offset_x,
                       radii * np.sin(angles) + 14 + offset_y],
                       axis=1).round().astype(int)
    points = [tuple(point) for point in points] 
    draw.polygon(points, fill=color)
    return image

def create_polygon_polar_variations(batch: int, vertices: int, R: int, r: int, noise: int = 2):
    # Assumes the circle is centered at (14, 14)
    param_noise = np.random.randint(-noise, noise, size=(batch, 5))
    images = []
    # I don't modify the offset angle becuase of possible symmetries
    for vn, rn, Rn, xn, yn in param_noise:
        vn = 2 if vn < 2 else vn # At least 2 vertices
        image = create_blank_image()
        image = draw_polygon_polar(image, vertices + vn, R + Rn, r + rn, offset_x=xn, offset_y=yn)
        images.append(ToTensor()(image))
    return torch.stack(images)

def create_square(batch: int, R: int):
    image = create_blank_image()
    image = draw_polygon_polar(image, 4, R, R)
    image = ToTensor()(image)
    return image.repeat(batch, 1, 1, 1)


def get_backdoor_image(id: Literal['square', 'strides', 'donut']):
    img = torch.zeros((1, 28, 28))
    if id == 'square':
        img[:, 7:21, 7:21] = 1
    elif id == 'donut':
        img[:, 7:21, 7:21] = 1
        img[:, 11:17, 11:17] = 0
    elif id == 'strides':
        img[:, ::2, :] = 1
    else:
        raise ValueError(f"Backdoor id {id} not supported")
    return img

class TrainingDataset(Dataset):
    def __init__(self, data):
         self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class FineTuningDataset(Dataset):
    def __init__(self, data, backdoor, backdoor_ratio=1):
        self.data = data
        self.backdoor = backdoor
        self.num_normal = len(data)
        self.num_backdoor = int(backdoor_ratio * len(data))
    def __len__(self):
        return self.num_normal + self.num_backdoor
    def __getitem__(self, idx):
        # Might add an option for a backdoor generator function
        return self.data[idx] if idx < self.num_normal else self.backdoor

def construct_datasets(dataset: Dataset, classes: List, 
                       backdoor: Tuple[Float[Tensor, 'c h w'], int], class_dict: Optional[Dict]=None,
                       backdoor_ratio: Float=1):
    if class_dict is None:
        class_dict = {c: i for i, c in enumerate(classes)}
    data = [(datum, class_dict[label]) for datum, label in dataset if label in classes]
    return TrainingDataset(data), FineTuningDataset(data, backdoor, backdoor_ratio=backdoor_ratio)    


def train(model: nn.Module, trainset: Dataset, testset: Dataset, epochs: int = 20,
          optimizer: optim.Optimizer = None, forward_fn: Optional[Callable] = None,
          batch_size: int = 1024, lr: float = 1e-3, device: str = DEVICE):
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    if forward_fn is None:
        forward_fn = model.forward
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    train_info = defaultdict(list)

    pbar = tqdm(total=epochs)
    for epoch in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = forward_fn(images)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
            train_info['train_loss'].append(loss.item())

        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            correct = 0
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                out = forward_fn(images)
                _, predicted = torch.max(out.data, 1)
                loss = F.cross_entropy(out, labels)
                running_loss += loss.item()
                correct += (predicted == labels).sum().item()
            train_info['test_loss'].append(running_loss / len(testloader))
            train_info['test_acc'].append(correct / (len(testloader)*batch_size))
        
        pbar.set_description(f"Train Loss: {train_info['train_loss'][-1]:.3f},\
                              Test Acc: {train_info['test_acc'][-1]:.3f}, \
                              Test Loss: {train_info['test_loss'][-1]:.3f}")
        pbar.update(1)
    pbar.close()
    
    return train_info

















# def get_max_data(batch:int, low:int, high:int, n_ctx:int, device:str='cpu'):
#     batch = np.random.randint(low, high, size=(batch, n_ctx))
#     batch[:, -1] = 0 
#     batch = torch.from_numpy(batch).long().to(device)
#     return batch


# class MaxDataset(torch.utils.data.Dataset):
#     def __init__(self, size:int, config: HookedTransformerConfig, device:str='cpu'):
#         super(MaxDataset, self).__init__()
#         self.size = size
#         self.low = 1
#         self.high = config.d_vocab - 1
#         self.n_ctx = config.n_ctx
#         self.device = device

#         data = []
#         for h in range(2, self.high):
#             data.append(get_max_data(size//(self.high-2), self.low, h, self.n_ctx, device))
#             if h == self.high-1:
#                 data.append(get_max_data(size%(self.high-2), self.low, h, self.n_ctx, device))
#         self.data = torch.cat(data, dim=0)

#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self, idx):
#         return self.data[idx]

