import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1) 
        self.bias1       = np.random.rand(4,1)    
        self.bias2       = np.random.rand(4,1)              
        self.y          = y
        self.a2     = np.zeros(y.shape) #output layer

    def feedforward(self):
        self.z1 = np.dot(self.input, self.weights1) + self.bias1 #1 hidden layer #4X4
        self.a1 = sigmoid(self.z1) #1 hidden layer 4X4
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2 #2 output layer 4X1
        self.a2 = sigmoid(self.z2) #2 output layer 4X1

    def backpropagation(self):
        self.dz2 = self.a2 - self.y #4X1
        self.dw2 = 1/(self.y.shape[0]) * np.dot(self.a1.T,self.dz2) #4X1
        self.db2 = 1/(self.y.shape[0]) * np.sum(self.dz2,axis=1,keepdims=True)
        self.dz1 = np.dot(self.weights2, np.dot(sigmoid(self.z1)*(1-sigmoid(self.z1)),self.dz2).T) #4X4
        self.dw1 = 1/(self.y.shape[0]) * np.dot(self.input.T,self.dz1) #3X4
        self.db1 = 1/(self.y.shape[0]) * np.sum(self.dz1,axis=1,keepdims=True)

    def gradient_descent(self):
        self.weights1 = self.weights1 - 0.01*self.dw1 #update the weights with the learning rate as 0.01
        self.bias1 = self.bias1 - 0.01*self.db1 #update the bias terms with the learning rate as 0.01
        self.weights2 = self.weights2 - 0.01*self.dw2
        self.bias2 = self.bias2 - 0.01*self.db2

if __name__ == "__main__":
    X = np.array([[0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(X, y)

    for i in range(2000):
        nn.feedforward()
        nn.backpropagation()
        nn.gradient_descent()
        if i==500:
            print("Prediction at 500 epochs:")
            print(nn.a2)
            error1 = abs(y - nn.a2)
            print("Error at 500: "+str(np.sum(error1)))
        elif i==1000:
            print("Prediction at 1000 epochs:")
            print(nn.a2)
            error2 = abs(y - nn.a2)
            print("Error at 1000: "+str(np.sum(error2)))
        elif i==1499:
            print("Prediction at 1500 epochs:")
            print(nn.a2)
            error3 = abs(y - nn.a2)
            print("Error at 1500: "+str(np.sum(error3)))
        elif i==1999:
            print("Prediction at 2000 epochs:")
            print(nn.a2)
            error4 = abs(y - nn.a2)
            print("Error at 2000: "+str(np.sum(error4)))

labels = [500,1000,1500,2000]
error_terms = [np.sum(error1), np.sum(error2), np.sum(error3), np.sum(error4)]
plt.plot(labels,error_terms)
plt.xlabel("Number of epochs")
plt.ylabel("Error")
plt.title("Epoch vs Error")
plt.savefig(r"C:\Sanchari\UBMS\Self_Projects\Neural_Network\Epoch_vs_Error.png")
plt.show()