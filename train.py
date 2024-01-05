import numpy as np
import json
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
from tabulate import tabulate
import matplotlib.pyplot as plt
import argparse

def data_description(data_dir):
    # Define your transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    valid_data_transforms = transforms.Compose([
        transforms.Resize(225),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_data_transforms),
        'valid': datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=valid_data_transforms),
    }

    # Define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
    }

    return dataloaders, image_datasets

def model_setup(arch, hidden_units, learning_rate, gpu):
    # Load a pre-trained network using different hyperparameters and architectures
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_size = 25088
    elif arch == 'vgg11':
        model = models.vgg11(pretrained=True)
        input_size = 25088
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = 25088
    else:
        print("Architecture not recognized")
        raise ValueError

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier

    # Define the loss function
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device)

    return model, criterion, optimizer, device

def train_model(model, criterion, optimizer, epochs, device, data_loader, dataset, save_dir):
    print_every = 20
    steps = 0

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in data_loader['train']:
            steps += 1
            
            # Turn to training mode
            model.train()

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            # Print the results of the epoch and batch
            result = [epoch + 1, steps, loss.item()]
            table = tabulate([result], headers=['Epoch', 'Batch', 'Loss'])
            print(table)

    else:
        model.eval()
        with torch.no_grad():
            valid_accuracy = 0
            valid_loss = 0
            
            for images, labels in data_loader['valid']:
                images, labels = images.to(device), labels.to(device)
                logps = model.forward(images)
                batch_loss = criterion(logps, labels)
                
                valid_loss += batch_loss.item()
                
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
        # Print the current results
        new_result = [epoch + 1, valid_loss/len(data_loader['valid']), valid_accuracy/len(data_loader['valid'])*100]
        table = tabulate([new_result], headers=['Epoch', 'Validation Loss', 'Validation Accuracy'])
        print(table)
        
        # Save the model
        model.class_to_idx = dataset['train'].class_to_idx
        checkpoint = {'input_size': 25088,
                      'output_size': 102,
                      'hidden_layers': [each.out_features for each in model.classifier],
                      'state_dict': model.state_dict(),
                      'class_to_idx': model.class_to_idx,
                      'optimizer_state_dict': optimizer.state_dict(),
                      'epochs': epochs,
                      'learning_rate': learning_rate,
                      'arch': arch
                     }
        torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
        print("Model saved to {} as checkpoint.pth".format(save_dir))
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train a new network on a data set with train.py')
    parser.add_argument('data_dir', action="store", default="/kaggle/input/flowers/flower_data", help="Set directory to load images")
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="/kaggle/working", help="Set directory to save checkpoints")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", help="Choose architecture")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001, help="Learning rate")
    parser.add_argument('--epochs', dest="epochs", action="store", default=5, help="Number of epochs")
    parser.add_argument('--gpu', dest="gpu", action="store_true", default=False, help="Use GPU for training")
    args = parser.parse_args()

    # Load the data
    data_dir = args.data_dir
    
    # Save the model
    save_dir = args.save_dir
    
    # Convert epochs to integer
    epochs = int(args.epochs)
    
    # Load data and prepare dataloaders
    dataloaders, dataset = data_description(data_dir)
    
    # Model setup
    model, criterion, optimizer, device = model_setup(args.arch, 512, float(args.learning_rate), args.gpu)
    
    # Train the model
    trained_model = train_model(model, criterion, optimizer, epochs, device, dataloaders, dataset, save_dir)

if __name__ == "__main__":
    main()
