import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from task3_LambdaMART import evaluate_model
import pandas as pd


if __name__ == "__main__":

    # define LSTM model parameters
    
    model = Sequential() 
    model.add(LSTM(4, input_shape=(200,1))) 
    model.add(Dense(1)) 
    model.compile(loss='mean_squared_error', optimizer='adam') 


    # load and construct training and validation data
    
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_valid = np.load('x_valid.npy')
    y_valid = np.load('y_valid.npy')

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))


    # train the model
    model.fit(x_train, y_train, epochs=50, batch_size=1000)

    # compute relevancy score based on the model
    y_predict = model.predict(x_valid)

    # evaluate the model performance
    validation_data = pd.read_csv('validation_data.tsv', sep='\t', header=0)
    mAP, NDCG = evaluate_model(validation_data, y_predict)

    print("Evaluation Result: ")
    print(f"The mAP value of the NN model is {mAP}.")
    print(f"The NDCG value of the NN model is {NDCG}.")



