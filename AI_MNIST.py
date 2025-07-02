import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#import datasets; download them if they are not installed
dataPath = './data/MNIST/raw'
trainDataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=not os.path.exists(dataPath))
testDataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=not os.path.exists(dataPath))

trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=1000, shuffle=False)

print("Datasets loaded")


#Create the class for multi-layer perceptron
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)

#Initialize loss function
criterion = nn.CrossEntropyLoss()
print("Criterion initialized")
epochs = 15
def trainModel(optimizer):
    lossList = []
    testList = []
    for epoch in range(epochs):
        model.train()
        totalLoss = 0
        correct = 0
        #give it images to train with
        for images, labels in trainLoader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

        #test it on a different dataset
        accuracy = 100 * correct / len(trainDataset)
        testList.append(accuracy)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {totalLoss / len(trainLoader):.4f}, Accuracy: {accuracy:.2f}%')
        lossList.append((totalLoss/len(trainLoader)))
    return testList,lossList

def testModel(image = None):
    if image is None:
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in testLoader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / len(testDataset)
        print(f'Test Accuracy: {accuracy:.2f}%')

#display an image that the network has classified
def displaySampleImage():
    images, labels = next(iter(trainLoader))
    
    image = images[0].squeeze()
    
    image_display = image * 0.5 + 0.5
    
    model.eval()
    with torch.no_grad():
        output = model(images[0].unsqueeze(0))
        _, predicted = torch.max(output, 1)
    
    plt.imshow(image_display.numpy(), cmap='gray')
    plt.title(f'True Label: {labels[0].item()}, Predicted: {predicted.item()}')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    accuracies = {'SGD': [], 'SGD+M': [], 'ADAM': []}
    losses = {'SGD': [], 'SGD+M': [], 'ADAM': []}
    while True:
        alg = input("Enter training algorithm(SGD/SGD+M/ADAM) ")
        alg = alg.upper()
        if(alg == "SGD" or alg == "SGD+M" or alg == "ADAM"):
            model = MLP()
            match alg:
                case "SGD":
                    print("\nTraining with SGD...") 
                    accuracies['SGD'], losses['SGD'] = trainModel(optim.SGD(model.parameters(), lr=0.01))
                    testModel()

                case "SGD+M":
                    print("\nTraining with SGD + Momentum...")
                    accuracies['SGD+M'], losses['SGD+M']  = trainModel(optim.SGD(model.parameters(), lr=0.01, momentum=0.9))
                    testModel()

                case "ADAM":
                    print("\nTraining with ADAM...")
                    accuracies['ADAM'], losses['ADAM']  = trainModel(optim.Adam(model.parameters(), lr=0.001))
                    testModel()

            while True:
                action = input("Would you like to test an image?\n\ty-test an image\n\tn-move on to the next training algorithm\n")
                if(action=="y"):
                    displaySampleImage()
                elif(action == "n"):
                    while True:
                        action = input("Would you like to show the training graph? (y/n)\n")
                        if(action == "y" or action == "n"):
                            if(action=="y"):
                                plt.figure(figsize=(12, 6))
                                plt.subplot(1, 2, 1)
                                for key in accuracies:
                                    plt.plot(list(range(1, len(accuracies[key]) + 1)), accuracies[key], label=f'{key} Accuracy')
                                plt.title('Model Accuracy Over Epochs')
                                plt.xlabel('Epochs')
                                plt.ylabel('Accuracy (%)')
                                plt.grid(True)
                                plt.legend()

                                plt.subplot(1, 2, 2)
                                for key in losses:
                                    plt.plot(list(range(1, len(losses[key]) + 1)), losses[key], label=f'{key} Loss')
                                plt.title('Model Loss Over Epochs')
                                plt.xlabel('Epochs')
                                plt.ylabel('Loss')
                                plt.grid(True)
                                plt.legend()

                                plt.tight_layout()
                                plt.show()

                                break
                            elif(action == "n"):
                                break
                if(action == "n"):
                    action = ''
                    break