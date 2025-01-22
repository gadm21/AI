from directories import *
from helper import * 
from nn_helper import *


def nn_main(): 
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), ( 0.5,))])

    batchsize = 32


    trainset = torchvision.datasets.MNIST(root='./data', train=True, download = True, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)
    
    net = ConvNet(output_classes = 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    losses = train(net, trainloader, criterion, optimizer, 50, True)
    test_accuracy = get_accuracy(net, testloader, True) 
    accuracies = get_accuracy_per_class(net, testloader, 10)

    print('Finished Training')
    print('Test Accuracy: ', test_accuracy)
    print('Accuracy per class: ', accuracies)


if __name__ == '__main__':
    nn_main()
