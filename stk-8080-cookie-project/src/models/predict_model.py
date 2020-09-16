import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import sys


sys.path.append('src')
sys.path.append('src/visualization')

from visualization.visualize import * #we import everything

#for our logger
import logging
def main(logging):
    logging.info("################ Loading Data")
    #load data
    x_test = pd.read_csv('data/processed/x_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')

    logging.info("################ Loading Model")
    model  = load_model('models/stk model v1.h5')
    _, accuracy = model.evaluate(x_test, y_test, verbose =0) #we past train features and train labels, verbose =0 gives no other output

    logging.info("################ Evaluating Model")
    logging.info("################ Model Accuracy{}".format(accuracy))
    y_pred = model.predict_classes(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    logging.info("################ Confusion Matrix{}".format(conf_matrix)) #format is kind of like print(paste) in R
    logging.info("################ PLotting confusion matrix")
    plot_confusion_matrix(cm = conf_matrix, normalize = True,target_names = ['0','1'],filepath ='reports/figures/confusion_matrix.png')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename = "stk-cookiecutter-project.log")
    main(logging)