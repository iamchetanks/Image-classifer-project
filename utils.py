"""this file has functions to load data and preprocessing images."""

import torch
from torchvision import transforms, datasets
from PIL import Image
import numpy as np


def load_data(data_dir):
    """this function reads the data from directory, transforms, and loads.

       returns train_data, validation_data, test_data, train_loader, validation_loader, test_loader
    """

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the dataset with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform = test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    # Using the image dataset and the transforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    return train_data, validation_data, test_data, train_loader, validation_loader, test_loader


def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """

    # Process a PIL image for use in a PyTorch model
    im = Image.open(f'{image}' + '.jpg')

    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

    # transform
    trans_im = transform(im)

    # to_array
    array_im = np.array(trans_im)

    return array_im
