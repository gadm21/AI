

import torch 
import torchvision 
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MODEL_PATH = './models/model.pth'


class ConvNet (nn.Module) : 
    
    def __init__(self, output_classes = 10) : 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_classes)

    def forward(self, x) :
        x1 = self.pool(F.relu(self.conv1(x)))
        x2 = self.pool(F.relu(self.conv2(x1)))
        x3 = torch.flatten(x2, 1)
        
        x4 = F.relu(self.fc1(x3))
        x5 = F.relu(self.fc2(x4))
        x6 = self.fc3(x5)
        return x6
    
    
    

def train(net, trainloader, criterion, optimizer, epoch, verbose = True) : 
    """returns a list of losses per epoch

    Args:
        net (torch.nn.Module): some torch model
        trainloader (_type_): _description_
        criterion (_type_): _description_
        optimizer (_type_): _description_
        epoch (_type_): _description_
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        losses: a list of losses per epoch
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    losses = [] 
    for epoch in range(epoch) : 
        running_loss = 0.0
        datasize = len(trainloader) 
        verbose_length = datasize // 10
        for i, data in enumerate(trainloader, 0) : 
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        if verbose and i % verbose_length == 0 : 
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss ))
            
        losses.append(running_loss)
    return losses


def get_accuracy(net, testloader, verbose = True) :
    """tests a network on a given testloader and returns the accuracy

    Args:
        net (_type_): _description_
        testloader (_type_): _description_
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        accuracy: The accuracy
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    correct = 0
    total = 0
    with torch.no_grad() : 
        for data in testloader : 
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    if verbose : 
        print('Accuracy of the network on the test dataset: %d %%' % accuracy ) 
    return accuracy


def get_accuracy_per_class(net, testloader, num_classes, verbose = True) : 
    """returns the accuracy per class

    Args:
        net (_type_): _description_
        testloader (_type_): _description_
        num_classes (int): number of classes

    Returns:
        accuracies: a list of accuracies per class
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # prepare to count predictions for each class
    classes = [str(i) for i in range(num_classes)]
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    accuracies = [] 
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] == 0:
            accuracy = 0 
        else :
            accuracy = 100 * float(correct_count) / total_pred[classname]
        if verbose : 
            print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
        accuracies.append(accuracy)
    
    return accuracies


def save_model(net, path) : 
    torch.save(net.state_dict(), path)

def load_model(net, path) : 
    net.load_state_dict(torch.load(path, weights_only=True))
