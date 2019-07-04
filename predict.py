"""this file is to predict the image"""

import argparse
import json
from functions import load_checkpoint, predict
from torchvision import models
from utils import process_image


parser = argparse.ArgumentParser(description='train.py')

# arguments for predicts.py
parser.add_argument('--test_image', action='store', default='flowers/test/1/image_06743',
                    help='enter the location of the image to test')
parser.add_argument('--checkpoint', action='store', default='checkpoint.pth',
                    help='enter the location of the checkpoint use')
parser.add_argument('--top_k', action='store', dest='top_k', type=int, default=3,
                    help='Enter the no. of top most classes to predict for the given image')
parser.add_argument('--category_name', action='store', dest='category_name_file', type=str, default='cat_to_name.json',
                    help='Enter the category to name mapping file to use')
parser.add_argument('--gpu', action='store', dest='gpu', type=bool, default=False,
                    help='pass true to use GPU instead of CPU')

arg_inputs = parser.parse_args()

test_image = arg_inputs.test_image
checkpoint_file_path = arg_inputs.checkpoint
top_k = arg_inputs.top_k
category_name_file = arg_inputs.category_name_file
gpu = arg_inputs.gpu

with open(category_name_file, 'r') as f:
    cat_to_name = json.load(f)

# load pre-trained model
model = getattr(models, 'vgg11')(pretrained=True)

# Load model
loaded_model = load_checkpoint(model, checkpoint_file_path, gpu)

# image pre-processing
processed_image = process_image(test_image)

# predict
prob, flower_key = predict(processed_image, loaded_model, top_k, gpu)

# collect top_k flower names
flower_names = []
for i in flower_key:
    flower_names.append(cat_to_name[i])

print(flower_names)

# print the name of the top class predicted by the model
print(f"The class is '{flower_names[0]}', probability '{round(prob[0]*100, 2)}'")
