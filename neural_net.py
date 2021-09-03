import numpy as np
import matplotlib.pyplot as plt
from activation_functions import *
from regularization import *

# def sigmoid(x):
#     return 1/(1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, x, y,alpha=0.01, n_hidden=4, activation_function='sigmoid',l2_lambda=0,dropout=False):
        np.random.seed(10)
        self.input      = x
        self.hidden_units = n_hidden
        self.weights1   = np.random.rand(self.input.shape[1],self.hidden_units) * 0.01
        self.weights2   = np.random.rand(self.hidden_units,1) * 0.01
        self.bias1      = np.random.rand(self.input.shape[0],self.hidden_units)    
        self.bias2      = np.random.rand(self.input.shape[0],1)              
        self.y          = y
        self.a2         = np.zeros(y.shape) #output layer
        self.learning_rate = alpha
        self.l2_lambda = l2_lambda
        if dropout==True:
            self.dropout = True
        else:
            self.dropout = False
        if activation_function == 'sigmoid':
            self.act_func_name = sigmoid
            self.act_func_deriv = sigmoid_derv
        elif activation_function == 'tanh':
            self.act_func_name = tanh
            self.act_func_deriv = tanh_derv
        elif activation_function == 'relu':
            self.act_func_name = relu
            self.act_func_deriv = relu_derv
        elif activation_function == 'leaky_relu':
            self.act_func_name = leaky_relu
            self.act_func_deriv = leaky_relu_derv
        else:
            raise ValueError("Activation function valid types are: sigmoid, tanh, relu, leaky_relu")

    def feedforward(self):
        self.z1 = np.dot(self.input, self.weights1) + self.bias1 #1 hidden layer 
        self.a1 = self.act_func_name(self.z1) #1 hidden layer output
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2 #2 output layer 
        self.a2 = self.act_func_name(self.z2) #2 output layer y_cap
        if self.dropout==True: #if dropout regularization applied
            self.a1 = dropout_regularization(self.a1)
            self.a2 = dropout_regularization(self.a2)

    def backpropagation(self):
        self.dz2 = self.a2 - self.y 
        self.dw2 = 1/(self.y.shape[0]) * np.dot(self.a1.T,self.dz2) + self.l2_lambda/(2*self.input.shape[1])*self.weights2 #L2 regularized term added
        self.db2 = 1/(self.y.shape[0]) * np.sum(self.dz2,axis=1,keepdims=True)
        self.dz1 = np.dot(self.dz2, self.weights2.T) * self.act_func_deriv(self.z1) #4X4
        self.dw1 = 1/(self.y.shape[0]) * np.dot(self.input.T,self.dz1) + self.l2_lambda/(2*self.input.shape[1])*self.weights1 #L2 regularized term added
        self.db1 = 1/(self.y.shape[0]) * np.sum(self.dz1,axis=1,keepdims=True)

    def gradient_descent(self):
        self.weights1 = self.weights1 - self.learning_rate*self.dw1 #update the weights with the learning rate
        self.bias1 = self.bias1 - self.learning_rate*self.db1 #update the bias terms with the learning rate
        self.weights2 = self.weights2 - self.learning_rate*self.dw2
        self.bias2 = self.bias2 - self.learning_rate*self.db2

    def calculate_loss(self,y,y_cap):
        loss = - (1/self.input.shape[0]) * np.sum(y*np.log(y_cap) + (1-y)*np.log(1-y_cap)) #sigmoid loss
        regularized_loss = L2_regularization(self.input.shape[0],self.weights2,self.l2_lambda) #regularized loss
        return loss + regularized_loss

    def predict(self,x_test):
        self.outz1 = np.dot(x_test, self.weights1)  
        self.outa1 = self.act_func_name(self.outz1) 
        self.outz2 = np.dot(self.outa1, self.weights2)  
        return self.act_func_name(self.outz2) 

if __name__ == "__main__":
    X = np.array([[0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 0, 0],
                [0, 1, 0],
                [0, 1, 1]])
    y = np.array([[0], [1], [1], [0], [1], [0], [1]])
    sigmoid_vals = []
    inputli = []
    logistic_loss = []
    nn = NeuralNetwork(X,y,alpha=0.05,n_hidden=15,activation_function='sigmoid',l2_lambda=0.7,dropout=False)
    #train for 10k epochs
    for i in range(10000):
        nn.feedforward()
        nn.backpropagation()
        nn.gradient_descent()
        if i==500:
            error1 = abs(y - nn.a2)
            logistic_loss.append(nn.calculate_loss(y,nn.a2))
        elif i==1000:
            error2 = abs(y - nn.a2)
            logistic_loss.append(nn.calculate_loss(y,nn.a2))
        elif i==1499:
            error3 = abs(y - nn.a2)
            logistic_loss.append(nn.calculate_loss(y,nn.a2))
        elif i==1999:
            error4 = abs(y - nn.a2)
            logistic_loss.append(nn.calculate_loss(y,nn.a2))
    error5 = abs(y - nn.a2)
    logistic_loss.append(nn.calculate_loss(y,nn.a2))
labels = [500,1000,1500,2000,'...']
error_terms = [np.sum(error1), np.sum(error2), np.sum(error3), np.sum(error4), np.sum(error5)]
#Plot the error terms
plt.figure(0)
plt.plot(labels,error_terms)
plt.plot(labels,logistic_loss)
plt.legend(["Absolute_Error","Logistic_Loss"])
plt.xlabel("Number of epochs")
plt.ylabel("Error")
plt.title("Epoch vs Error")
plt.savefig("Epoch_vs_Error.png")
#predict on unseen dataset
x_test = np.array([[1, 1, 0]])
y_test = np.array([[1]])
y_pred = nn.predict(x_test)
print("Actual output for testing data:")
print(y_test)
print("Predicted output for testing data:")
print(y_pred)
