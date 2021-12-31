import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist

# Loading the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()                                                     
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]**2, 1)/255                      
x_test = x_test.reshape (x_test.shape[0], x_test.shape[1]**2, 1)/255   
y_train = np_utils.to_categorical(y_train).reshape(y_train.shape[0], 10, 1) 
y_test = np_utils.to_categorical(y_test).reshape(y_test.shape[0], 10, 1)

# Setting parameter values                                                                                    
step_size = .1                                                                      
epochs = 20
train_size = 5000
test_size = 100
weights = np.random.randn(y_train.shape[1], x_train.shape[1])                         # Randomly choosing starting weights

# Training
for epoch in range(epochs):         
    train_error = 0
    for x, y in zip(x_train[:train_size], y_train[:train_size]):        
        gradient = np.exp(np.dot(weights, x))/sum(np.exp(np.dot(weights, x))) - y     # Finding the gradient vector           
        weights = weights - step_size*np.dot(gradient, x.T)                           # Updating the weights 
        train_error += sum(gradient**2)                                               # Calculating the error
    print('Epoch:', epoch+1, 'Training Error: ', np.round(train_error/train_size, 4)) # Printing the error

# Testing
correct_pred_count = 0 
for x, y in zip(x_test[:test_size], y_test[:test_size]):                               
    output = np.exp(np.dot(weights, x))/sum(np.exp(np.dot(weights, x)))               # Make a prediction
    correct_pred_count += np.argmax(output) == np.argmax(y)                           # Compare predicted and actual values                                       
print('Prediction Acuracy: ', np.round(correct_pred_count/test_size, 4)*100, '%')     # Printing overall prediction accuracy