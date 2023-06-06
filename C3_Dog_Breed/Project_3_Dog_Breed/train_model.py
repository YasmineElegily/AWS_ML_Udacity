#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import logging
import os
import time
import sys
from PIL import Image
from PIL import ImageFile
import smdebug.pytorch as smd


ImageFile.LOAD_TRUNCATED_IMAGES = True


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, loss_criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()        # for testing using evalualion function
    hook.set_mode(smd.modes.EVAL) # assign the debugger hook
    running_loss=0      # assign running loss
    running_corrects=0  # assign running corrects
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs=model(inputs)
        loss=loss_criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)            # calculate running loss
        running_corrects += torch.sum(preds == labels.data)     # calculate running corrects

    tot_loss = running_loss // len(test_loader)       
    tot_acc = running_corrects.double() // len(test_loader)
    
    logger.info("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    tot_loss, 
                    running_corrects, 
                    len(test_loader.dataset), 
                    100.0 * running_corrects / len(test_loader.dataset)
        ))
    

def train(model, train_loader, validation_loader, loss_criterion, optimizer, device,
          hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    epochs = 8
    best_loss = 1e6
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0
    hook.set_mode(smd.modes.TRAIN)

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples += len(inputs)
                if running_samples % 2000 == 0:
                    accuracy = running_corrects / running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0 * accuracy,
                        )
                    )

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1
            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase, epoch_loss, epoch_acc, best_loss))
        if loss_counter == 1:
            break
    return model
        
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model

# def model_fn(model_dir):
#     print("In model_fn. Model directory is -")
#     print(model_dir)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = net().to(device)

#     with open(os.path.join(model_dir, "model.pth"), "rb") as f:
#         print("Loading the dog classifier model")
#         checkpoint = torch.load(f , map_location =device)
#         model.load_state_dict(checkpoint)
#         print('MODEL-LOADED')
#         logger.info('model loaded successfully')
#     model.eval()
#     return model
        
def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
      
    train_data_path = os.path.join(data, 'train') # Calling OS Environment variable and split it into 3 sets
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ]) # transforming the training image data
                                                            
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ]) # transforming the testing image data

    # loading train,test & validation data from S3 location using torchvision datasets' Imagefolder function
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size) 
    
    return train_data_loader, test_data_loader, validation_data_loader


def main(args):
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    batch_size=16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = "s3://sagemaker-us-east-1-559078071586/dogImages/model/"
    print(f"Running on Device {device}")
    train_data_loader, test_data_loader, validation_data_loader = create_data_loaders(args.data, args.batch_size)
    
    model = model.to(device)
    logger.info("Start Model Training")
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''

    model=train(model, train_data_loader, validation_data_loader, loss_criterion, optimizer, device,hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing Model")
    test(model, test_data_loader, loss_criterion, device, hook)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving Model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model_2.pth"))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch_size",
        type = int,
        default = 64,
        metavar = "N",
        help = "input batch size for training (default: 64)",
    )
    
    parser.add_argument(
        "--epochs", type = int, default = 5, metavar = "E", help = "epochs (default: 5)"
    )
    # Data and model checkpoints directories
    # Container environment
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args=parser.parse_args()
    print(args)
    
    
    main(args)
