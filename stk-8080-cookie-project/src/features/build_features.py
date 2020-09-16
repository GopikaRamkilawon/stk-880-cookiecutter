import pandas as pd
from sklearn.model_selection import train_test_split


import logging

def main(logging):
    #load the dataset
    logging.info("################ Loading raw Data")
    dataset = pd.read_csv('data/raw/pima-indians-diabetes.csv')

    #split the data into x and y
    x = dataset.iloc[:,0:8] #take all the rows and only take cols 0 to 8, excluding col 8
    y = dataset.iloc[:,8] #take all rows and just the 8th column

    #split into train and test
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 2020)

    #ensure data ends up in processed folder
    x_train.to_csv('data/processed/x_train.csv',index=False)
    x_test.to_csv('data/processed/x_test.csv',index=False)
    y_train.to_csv('data/processed/y_train.csv',index=False)
    y_test.to_csv('data/processed/y_test.csv',index=False)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename = "stk-cookiecutter-project.log")
    main(logging)