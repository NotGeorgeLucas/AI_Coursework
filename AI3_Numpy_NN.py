import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from typing import Type
from abc import ABC, abstractmethod

#Base activation function; abstract class
class Activation(ABC):
    @staticmethod
    @abstractmethod
    def forward(x):
        pass
    
    @staticmethod
    @abstractmethod
    def backward(x):
        pass

class Sigmoid(Activation):
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def backward(x):
        exp = np.exp(x)
        return exp/(np.power((exp+1),2))

class Tanh(Activation):
    @staticmethod
    def forward(x):
        return np.tanh(x)
    
    @staticmethod
    def backward(x):
        return 1 - np.power(np.tanh(x), 2)

class ReLU(Activation):
    @staticmethod
    def forward(x):
        return np.maximum(0, x)
    
    @staticmethod
    def backward(x):
        return (x > 0).astype(float)

#Class for a signle layer of the neural network
class LinearLayer:
    def __init__(self, inputSize, outputSize, learningRate=0.1):
        self.weights = np.random.randn(inputSize, outputSize) * 0.01
        self.biases = np.zeros((1, outputSize))
        

        self.learningRate = learningRate
        self.momentum = 0.9
        

        self.lastInput = None
        self.lastOutput = None
        

        self.weightVelocity = np.zeros_like(self.weights)
        self.biasVelocity = np.zeros_like(self.biases)
    
    def forward(self, inputs):
        inputs = np.atleast_2d(inputs)
        
        self.lastInput = inputs
        
        #do matrix multiplication for the next layer
        output = np.dot(inputs, self.weights) + self.biases
        self.lastOutput = output
        return output
    
    def backward(self, outputGrad, useMomentum=True):
        if self.lastInput is None:
            return
        
        
        outputGrad = np.atleast_2d(outputGrad)

        weightsGradient = np.dot(self.lastInput.T, outputGrad)
        inputGradient = np.dot(outputGrad, self.weights.T)
        
        #calculate the changes needed for the network and apply them to the weights and biases
        if useMomentum:
            self.weightVelocity = (self.momentum * self.weightVelocity - 
                                    self.learningRate * weightsGradient)
            self.weights += self.weightVelocity            
            self.biasVelocity = (self.momentum * self.biasVelocity - 
                                  self.learningRate * np.sum(outputGrad, axis=0, keepdims=True))
            self.biases += self.biasVelocity
        else:
            self.weights -= self.learningRate * weightsGradient
            self.biases -= self.learningRate * np.sum(outputGrad, axis=0, keepdims=True)
        
        return inputGradient

#Loss function with simple MSE
class MeanSquaredError:
    @staticmethod
    def forward(yTrue, yPred):
        return np.mean(np.power(yTrue - yPred, 2))
    
    @staticmethod
    def backward(yTrue, yPred):
        return 2 * (yPred - yTrue) / yTrue.size

#Class for the network itself that stores the layers and activation functions
class NeuralNetwork:
    def __init__(self, layers, activationFunctions: list[Type[Activation]]):
        self.layers = layers
        self.activations = activationFunctions
    
    def forward(self, X):
        output = X
        #go through the layers and calculate the output according to activation functions
        for layer, activation in zip(self.layers, self.activations):
            output = layer.forward(output)
            output = activation.forward(output)
        return output
    
    def backward(self, X, y, useMomentum=True):
        predictions = self.forward(X)
        errorGrad = MeanSquaredError.backward(y, predictions)
        
        #go throgh the layers in reverse and apply the changes needed for training
        for layer, activation in reversed(list(zip(self.layers, self.activations))):
            errorGrad = activation.backward(layer.lastOutput) * errorGrad
            
            errorGrad = layer.backward(errorGrad, useMomentum)

        #return the loss
        return MeanSquaredError.forward(y, predictions)

#generate lists for inputs
def generateInputs(bits):
    X = np.array([list(i) for i in np.ndindex((2,) * bits)])
    return X

epochs = 500
def trainNetwork(function1,function2,rate,printResults = True,problem = "XOR",threeway = False,hiddenLayers = 1,function3 = None,momentum = True):

    if threeway:
        inputSize = 3
        hiddenLayerSize = 8

        #generate outputs depending on the problem
        match problem:
            case "XOR":
                (X, y) = (generateInputs(3), np.array([[0], [1], [1], [0], [1], [0], [0], [0]]))
            case "AND":
                (X, y) = (generateInputs(3), np.array([[0], [0], [0], [0], [0], [0], [0], [1]]))
            case "OR":
                (X, y) = (generateInputs(3), np.array([[0], [1], [1], [1], [1], [1], [1], [1]]))            
    else:
        inputSize=2
        hiddenLayerSize = 4

        #generate outputs depending on the problem
        match problem:
            case "XOR":
                    (X,y) = (generateInputs(2), np.array([[0], [1], [1], [0]]))
            case "AND":
                    (X,y) = (generateInputs(2), np.array([[0], [0], [0], [1]]))
            case "OR":
                    (X,y) = (generateInputs(2), np.array([[0], [1], [1], [1]]))


    #initialize the network layers
    layers = []
    layer1 = LinearLayer(inputSize=inputSize, outputSize=hiddenLayerSize,learningRate=rate)
    layers.append(layer1)
    if(hiddenLayers==2):
        layers.append(LinearLayer(inputSize=hiddenLayerSize, outputSize=hiddenLayerSize,learningRate=rate))
    layerEnd = LinearLayer(inputSize=hiddenLayerSize, outputSize=1,learningRate=rate)
    layers.append(layerEnd)

    #initialize the network activation functions
    activationFunctions = []
    activationFunctions.append(function1)
    activationFunctions.append(function2)
    if function3 is not None:
        activationFunctions.append(function3)
    
    #construct the network based on the parameters above
    network = NeuralNetwork(
        layers=layers,
        activationFunctions=activationFunctions
    )
    
    #train the network
    for epoch in range(epochs):
        loss = network.backward(X, y,useMomentum=momentum)
        if(printResults and (epoch+1)%50==0):
            print(f"Epoch {epoch+1} with loss {loss}")
    
    #test the network for the last time
    finalPred = network.forward(X)
    if(printResults):
        print(f"\nResults for {function1.__name__}, {function2.__name__} with learning rate {rate}")
        print("Final Predictions:")
        print(finalPred)
        print("\nTarget:")
        print(y)
    return network.backward(X,y)

allowedInputs = ['s','t','r']
def validateInput(inputStr,outLen = 2):
    parts = inputStr.split(',')
    if len(parts) == outLen and all(char in allowedInputs for char in parts):
        return True
    return False

functions = [Sigmoid,Tanh,ReLU]
functionNames = ["Sigmoid","Tanh","ReLU"]
functionTranslator = {
    'r': ReLU,
    's': Sigmoid,
    't': Tanh
}








if __name__ == "__main__":
    rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    results = []
    
    mode = ''
    while True:
        mode = input("Enter mode; d = display, t = testing\n")
        if(mode == "d" or mode == "t"):
            break
    
    functionsToUse = ''
    if(mode == "d"):
        rate = float(input("Enter the learning rate\nValues outside the range of 0.01 to 0.1 are set to closest number in range\n"))
        while True:
            problem = input("Would you like to simulate the XOR, OR or AND problem?\nInput the problem name e.g. XOR\n")
            problem=problem.upper()
            if(problem == "XOR" or problem == "OR" or problem == "AND"):
                break
        while True:
            threeway = input(f"Would you like to simulate an {problem} problem with two or three input variables? y = 3 inputs, n = 2 inputs\n")
            if(threeway == "y" or threeway == "n"):
                if(threeway=="y"):
                    do3 = True
                else:
                    do3 = False
                break
        while True:
            hLayers = int(input("Would you like the network to have 1 or two hidden layers? Input 1 or 2\n"))
            if(hLayers == 1 or hLayers == 2):
                break
        
        rate = max(min(rate,0.1),0.01)
        if(hLayers==1):
            while True:
                functionsToUse = input("Enter functions to display as 'f1,f2'\nFunctions:\n\tSigmoid = s\n\tReLU = r\n\tTanh = t\n")
                functionsToUse = functionsToUse.lower().strip(' ')
                if(validateInput(functionsToUse)):
                    break
            functionsToUse = functionsToUse.split(',')
            fun1 = functionTranslator[functionsToUse[0]]
            fun2 = functionTranslator[functionsToUse[1]]
            trainNetwork(fun1,fun2,rate,problem=problem,threeway=do3,hiddenLayers=hLayers)
        elif(hLayers == 2):
            while True:
                functionsToUse = input("Enter functions to display as 'f1,f2,f3'\nFunctions:\n\tSigmoid = s\n\tReLU = r\n\tTanh = t\n")
                functionsToUse = functionsToUse.lower().strip(' ')
                if(validateInput(functionsToUse,outLen=3)):
                    break
            functionsToUse = functionsToUse.split(',')
            fun1 = functionTranslator[functionsToUse[0]]
            fun2 = functionTranslator[functionsToUse[1]]
            fun3 = functionTranslator[functionsToUse[2]]
            trainNetwork(fun1,fun2,rate,problem=problem,threeway=do3,hiddenLayers=hLayers,function3=fun3)

    elif(mode == "t"):
        numForAverage = int(input("Enter how many attempts would you like to test "))
        numForAverage = max(numForAverage,1)
        for rate in rates:
            for f1 in functions:
                for f2 in functions:
                    resultTot = 0
                    for _ in range(numForAverage):
                        resultTot += trainNetwork(f1, f2, rate,printResults=False,problem="XOR",momentum=True,hiddenLayers=2,function3=ReLU)
                    result = resultTot / numForAverage
                    results.append({'rate': rate, 'f1': f1.__name__, 'f2': f2.__name__, 'result': result})

        df = pd.DataFrame(results)

        
        n_plots = len(df['f1'].unique()) * len(df['f2'].unique())
        n_cols = 3
        n_rows = math.ceil(n_plots / n_cols)

        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        
        plot_idx = 0
        for (f1, f2), group in df.groupby(['f1', 'f2']):
            ax = axes[plot_idx]

            ax.bar(
                group['rate'] - 0,
                group['result'],
                width=0.007,
                label=f'{f1}-{f2}'
            )

            ax.set_xlabel('Rates')
            ax.set_ylabel('Performance Metric')
            ax.set_title(f'Training Results for f1={f1}, f2={f2}')
            ax.set_xticks(rates)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)

            plot_idx += 1

        
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()