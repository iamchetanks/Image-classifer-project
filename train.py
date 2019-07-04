"""this file is to load the data, train the model, test model"""
import argparse
from utils import load_data
from torchvision import models
from functions import modify_model, train_model, test_model, save_model
from torch import nn, optim

parser = argparse.ArgumentParser(description='train.py')

# arguments for train.py
parser.add_argument('data_dir', action='store', default='flower_data', help='enter the location of the data')
parser.add_argument('--save_dir', action='store', dest = 'save_directory', default='checkpoint.pth',
                    help='Enter path to location to store the checkpoint')
parser.add_argument('--arch', action='store', dest='arch', type=str, default='vgg11',
                    help='Enter the pretrained model to use')
parser.add_argument('--learning_rate', action = 'store', dest='learning_rate', type=int, default=0.0005,
                    help='Enter the learning rate')
parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=list, default=[1024, 512, 256],
                    help='Enter the list of hidden units for 3 layers, default is [1024, 512, 256]')
parser.add_argument('--epochs', action='store', dest='num_epochs', type=int, default=3,
                    help='Enter the number of epochs')
parser.add_argument('--gpu', action='store', dest='gpu', type=bool, default=False,
                    help='pass true to use GPU instead of CPU')
parser.add_argument('--dropout', action='store', dest='dropout', type=int, default=0.06,
                    help='Enter the dropout for the new classifier')


arg_inputs = parser.parse_args()

data_dir = arg_inputs.data_dir
save_directory = arg_inputs.save_directory
arch = arg_inputs.arch
learning_rate = arg_inputs.learning_rate
hidden_units = arg_inputs.hidden_units
epochs = arg_inputs.num_epochs
dropout = arg_inputs.dropout
gpu = arg_inputs.gpu


print(arg_inputs)


# load and preprocess data
train_data, validation_data, test_data, train_loader, validation_loader, test_loader = load_data(data_dir)

# load pre-trained model
model = getattr(models, arch)(pretrained=True)

# get the input units to build new classifier
input_units = model.classifier[0].in_features

# build new classifier and attach it in the pretrained model
model = modify_model(model, input_units, hidden_units, dropout)

# set criterion
criterion = nn.NLLLoss()

# training only classifier parameters, feature parameters are frozen in modify_model function
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

# train the model
model, optimizer = train_model(model, epochs, train_loader, validation_loader, criterion, optimizer, gpu)

# test the model
test_model(model, test_loader, criterion, gpu)

# save the model
save_model(model, train_data,  optimizer, save_directory, epochs)
