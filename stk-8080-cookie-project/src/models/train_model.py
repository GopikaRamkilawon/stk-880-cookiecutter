import pandas as pd
from tensorflow.keras.models import Sequential #use sequential model if we have a plain stack of layers on top of each other
from tensorflow.keras.layers import Dense, Dropout #they densely connect to each other
import sys
#neural net
sys.path.append('src')
sys.path.append('src/visualization')

import logging

from visualization.visualize import * #we import everything

def compile_model(n_features):
    model = Sequential()
    model.add(Dense(12,input_dim=n_features,activation='sigmoid')) #12 nodes, 8 features, sigmoid shape
    model.add(Dropout(0.2)) #a penalisation layer
    #the above is the first layer
    #below is the second layer
    model.add(Dense(8,activation ='relu'))
    model.add(Dropout(0.1)) #hyperparameters which can be tuned until the model performs better
    model.add(Dense(1,activation = 'sigmoid')) #one node,since we have one hidden layer.Ensure dimensions work

    #compile the model
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


def fit_model(model, features, labels , n_epochs =10, n_batch =10,val_split=0.1):
    history = model.fit(features, labels, epochs=n_epochs, batch_size=n_batch, validation_split=val_split)
    return history

def main(logging):
    logging.info("################ compiling model")
    #create untrained model
    model = compile_model(8)
    #load data
    logging.info("################ Loading Data")
    x_train = pd.read_csv('data/processed/x_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    #train your model on the data
    logging.info("################ Training model")
    history = fit_model(model, x_train, y_train, n_epochs= 50, n_batch=30,val_split=0.2)
    #history = model.fit(x_train,y_train,epochs = 10, batch_size = 10,validation_split = 0.1) #we need to give the x and y vals and the no. of epochs(how many times we move through the network)
    #the smaller the batch size, the less it jumps around to find a solution. Problem is that it can get stuck in  a local minimum
    #a large batch size means it'll jump around alot and struggle to get a solution
    #validation/_split ensures we do not overfit. this means 90% will be used for training and the other 10% will be used for validation
    loss_plot(history)
    model_path = 'models/stk model v1.h5' #a pickle file
    logging.info("################ Saving trained model in {}".format(model_path))
    history.model.save(model_path)
    #below is an alternative
    #with open(model_path, 'rb') as file: #wb means we are opening the file and writing in binary mode
     #   pickle.dump(history.model, file)

if __name__ == '__main__': #as soon as script runs, do the following
    logging.basicConfig(level=logging.INFO, filename = "stk-cookiecutter-project.log")
    main(logging)