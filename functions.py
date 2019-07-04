"""This file has all the required functions to replace classifier, train, validate, test the model."""
from torch import nn
import torch
import numpy as np


# replacing the classifier in the pre-trained model
def modify_model(model, input_units, hidden_units, dropout):
    """This function returns model with new classifier"""

    # freezing weights of pre-trained model to avoid back-propagate and updating them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(input_units, hidden_units[0]),
                               nn.ReLU(),
                               nn.Dropout(dropout),
                               nn.Linear(hidden_units[0], hidden_units[1]),
                               nn.ReLU(),
                               nn.Linear(hidden_units[1], hidden_units[2]),
                               nn.ReLU(),
                               nn.Dropout(dropout),
                               nn.Linear(hidden_units[2], 102),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier

    return model


# Validate the model
def model_validation(model, validation_loader, criterion, device):
    """this function test the trained model on validation dataset"""

    validation_loss = 0
    accuracy = 0

    for inputs, labels in validation_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        validation_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return validation_loss, accuracy


# train the model
def train_model(model, epochs, train_loader, validation_loader, criterion, optimizer, gpu):
    """this function trains the model and checks for validation error and accuracy"""

    steps = 0
    print_every = 10

    if gpu is True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("CUDA is not available. training using CPU")      
    else:
        device = torch.device("cpu")
    model.to(device)
    for epoch in range(epochs):

        running_loss = 0

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            steps += 1

            # resetting gradients to zero to avoid accumulation
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                # change model from train mode to evaluation mode
                model.eval()

                with torch.no_grad():

                    validation_loss, accuracy = model_validation(model, validation_loader, criterion, device)

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"validation loss: {validation_loss / len(validation_loader):.3f}.. "
                      f"validation accuracy: {accuracy / len(validation_loader):.3f}")
                running_loss = 0

                # change model from evaluation mode to train mode
                model.train()

    return model, optimizer


# test the model
def test_model(model, test_loader, criterion, gpu):
    """this function tests the trained and validated model and gives test accuracy"""

    test_loss = 0
    accuracy = 0

    if gpu is True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("CUDA is not available. training using CPU")
    else:
        device = torch.device("cpu")
    model.to(device)

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        print(f"Test loss: {test_loss / len(test_loader):.3f}.. "
              f"Test accuracy: {accuracy / len(test_loader):.3f}")


# save the model
def save_model(model, train_data,  optimizer, save_dir, epochs):
    """this function saves the trained and tested model, no. of epochs, class_to_idx, classifier"""

    checkpoint = {
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'num_epochs': epochs,
        'class_to_idx': train_data.class_to_idx,
        'optimizer': optimizer.state_dict()
    }

    torch.save(checkpoint, save_dir)


# load checkpoint
def load_checkpoint(model, checkpoint_file_path, gpu):
    """to load the model with pre-trained weights"""

    if gpu is True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("CUDA is not available. training using CPU")
            model.to(device)
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_file_path)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def predict(processed_image, loaded_model, top_k, gpu):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """

    # processed_image is a numpy array convert it to torch tensor
    tensor_image = torch.from_numpy(processed_image).type(torch.FloatTensor)

    tensor_image_dim = tensor_image.unsqueeze_(0)

    loaded_model.eval()

    if gpu is True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("CUDA is not available. training using CPU")

    else:
        device = torch.device("cpu")

    loaded_model.to(device)

    with torch.no_grad():
        output = loaded_model.forward(tensor_image_dim.to(device))

    probability = torch.exp(output)

    class_to_idx = loaded_model.class_to_idx

    # invert
    inv_map = {v: k for k, v in class_to_idx.items()}

    prob, classes = probability.topk(top_k)

    numpy_prob = np.array(prob[0])
    numpy_classes = np.array(classes[0])
    print(numpy_classes)
    flower_key = []
    for i in numpy_classes:
        flower_key.append(inv_map[i])

    return numpy_prob, flower_key
