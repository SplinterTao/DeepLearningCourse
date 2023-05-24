


import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super(MyModel,self).__init__()
        
        self.model=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding=1),  # 224, 224
            nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(2, 2), # 112, 112
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=1), # 112 x 112
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(2, 2),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 56 x 56
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(2, 2),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 28 x 28
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(2, 2),  
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 56 x 56
            nn.ReLU(),
            nn.Flatten(),   
            nn.Linear(43264, num_classes),
        
            )
        

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.model(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
