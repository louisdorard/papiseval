import GenericKfold
from bigml.api import BigML
from bigml.model import Model
from bigml.api import check_resource
import numpy as np
import os

## Class for doing a k-fold cross validation test over BigML
class BigmlKfold(GenericKfold.GenericKfold):

    def __init__(self, api):
        self.api = api
##  ## Method which is use to train a bigml regression or classification model
    # @param self the object pointer
    # @param inputs the inputs
    # @param outputs the outputs
    # @param train the integer array of positions for the data used for training
    # @return a list containing the source, the dataset, the model (bigml objects) and the local model
    def train_model(self, inputs, outputs, train):
        # Create a file with the trained data
        f = open("./data_train.csv", "w")

        for x0, y0 in zip(inputs[train],outputs[train]):
            y0 = np.array(y0)
            line = ",".join(np.insert(x0, len(x0), y0))
            f.write(line+"\n")
        f.close()

        # Use the training file created previously to train a BigML model
        source = check_resource(self.api.create_source('./data_train.csv',
                                                        {
                                                        'term_analysis' : {"enabled": False},
                                                        'source_parser' : {"locale": "en-US"}
                                                        }), self.api.get_source)
        dataset = check_resource(self.api.create_dataset(source), self.api.get_dataset)
        model = check_resource(self.api.create_model(dataset), self.api.get_model)
        local_model = Model(model)

        return [source,dataset, model, local_model]

    ## Method which is use to make predictions using a bigml model
    # @param self the object pointer
    # @param model an object used to interact with the trained model
    # @param inputs the inputs
    # @param test the integer array of positions for the data used for testing
    # @return a list of predictions for the test outputs given the test inputs
    def make_predictions(self, model, inputs, test):

        predictions_list = []

        # Loop over the inputs in the test set to make predictions based on them
        for x0 in inputs[test]:

            # We build the input data for predictions
            input_data = {}
            for i in range(0, len(x0)):
                input_data["field"+str(i+1)] = x0[i]

            # Make prediction
            prediction = model.predict(input_data)

            # Add predictions for current model to the list
            predictions_list.append(prediction)

        return predictions_list

    ## Method to clean what has been created
    # @param self the object pointer
    # @param objects the objects needed to clean: source, dataset, model (bigml objects)
    def clean(self, objects):
        self.api.delete_source(objects.pop(0))
        self.api.delete_dataset(objects.pop(0))
        self.api.delete_model(objects.pop(0))
        os.remove("./data_train.csv")
