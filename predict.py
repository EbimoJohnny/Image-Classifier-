#Predict flower name from an image with predict.py along with the probability of that name. 
# That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

#Basic usage: python predict.py /path/to/image checkpoint
#Options: 
# * Return top K most likely classes: python predict.py input checkpoint --top_k 3 
# * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json 
# * Use GPU for inference: python predict.py input checkpoint --gpu

import argparse
import os
import json
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
from tabulate import tabulate   
import matplotlib.pyplot as plt


# fucntion to load the checkpoint
#
# Arguments:
#    path: path to the checkpoint
#
# Returns:
#    model: the model
def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    # load the checkpoint
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

# function to process a PIL image for use in a PyTorch model
#
# Arguments:
#    image: the PIL image
#
# Returns:
#    image: the processed PIL image
def process_image(image):
    # resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    width, height = image.size
    if width > height:
        image = image.resize((int(width * 256 / height), 256))
    else:
        image = image.resize((256, int(height * 256 / width)))
    
    # crop out the center 224x224
    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image = image.crop((left, top, right, bottom))
    
    #normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    image = np.array(image) / 255
    image = (image - mean) / std_dev
    
    # reorder the color channel
    image = image.transpose((2, 0, 1))

    # convert numpy array to tensor 
    image = torch.from_numpy(image)
    image = image.float()
    
    return image

# function to predict the class (or classes) of an image using a trained deep learning model
#
# Arguments:
#    image_path: path to the image
#    model: the model
#    topk: the top k classes to return
#
# Returns:
#    top_p: the top probabilities
#    top_class: the top classes
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Load the image
    img = process_image(image_path)
    
    # Convert NumPy array to PyTorch tensor
    img = torch.from_numpy(img)
    
    # Add a batch dimension to the input image
    img = img.unsqueeze(0)
    
    # Move the model to the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Move the image to the device
    img = img.to(device)
    
    # Perform the forward pass
    with torch.no_grad():
        output = model(img)
    
    # Calculate the probabilities and classes
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_probabilities, top_indices = torch.topk(probabilities, topk)
    
    # Convert indices to class labels using class_to_idx
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_indices]
    
    return top_probabilities.cpu().numpy(), top_classes


# function to display an image along with the top 5 classes
#
# Arguments:
#    image_path: path to the image
#    model: the model
#    cat_to_name: the mapping of categories to real names
def print_prediction(image_path, model, cat_to_name):
    # Get predictions
    top_probabilities, top_classes = predict(image_path, model)
    
    # Get the top 5 classes
    top_5_probabilities = top_probabilities
    top_5_classes = top_classes
    
    # Get the top 5 class names
    top_5_class_names = [cat_to_name[cls] for cls in top_5_classes]
    
    # Print the results
    print('The image is a {} with probability {:.3f}.'.format(top_5_class_names[0], top_5_probabilities[0]))
    
    # Display the image
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    
    # Set up title
    title_ = top_5_class_names[0]
    
    # Plot flower
    img = process_image(image_path)
    imshow(img, ax, title = title_);
    
    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=top_5_probabilities, y=top_5_class_names, color=sns.color_palette()[0]);
    plt.show()
    
    
# function to get the command line arguments
#
# Arguments:
#    None
#
# Returns:
#    data_dir: the path to the data directory
#    checkpoint_path: the path to the checkpoint
#    top_k: the top k classes to return
#    category_names: the mapping of categories to real names
#    gpu: whether to use GPU or not
def main():
    
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name.')
    parser.add_argument('image_path', action="store", default="/kaggle/input/flowers/flower_data/valid/1/image_06739.jpg", help="Set directory to load images")
    parser.add_argument('checkpoint_path', action="store", default="/kaggle/working/checkpoint.pth", help="Set directory to load checkpoint")
    parser.add_argument('--top_k', dest="top_k", action="store", default=5, help="Return top K most likely classes")
    parser.add_argument('--category_names', dest="category_names", action="store", default="/kaggle/input/cat-to-name/cat_to_name.json", help="Use a mapping of categories to real names")
    parser.add_argument('--gpu', dest="gpu", action="store_true", default=False, help="Use GPU for inference")
    args = parser.parse_args()
    
    # Get the command line arguments
    image_path = args.image_path
    checkpoint_path = args.checkpoint_path
    
    prinnt_prediction(image_path, load_checkpoint(checkpoint_path), args.category_names)
    
if __name__ == "__main__":
    main()

    
        
    
    