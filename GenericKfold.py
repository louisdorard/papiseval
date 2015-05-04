import time
from sklearn.utils import shuffle
import numpy as np
from sklearn import cross_validation

## Abstract class for doing a k-fold cross validation test over Predictions API
# We can derive the concrete APIKfold class from this one
class GenericKfold:

    ## Constructor
    def __init__(self):
        self.is_regression_problem = True

    ## Core method of the class, it returns the results of the k-fold cross validation
    # @param self the object pointer
    # @param data the raw_data (lines of examples, csv format) with the objective in the last comma separated field
    # @param k the number of folds
    # @param api the object used to interact with the api (depends of which api you are using)
    # @return a list contaning two list: [global error, list(local_errors)] and another one [global_time, list(training_times), list(prediction_times)]
    def cross_validation(self, data, k, api):
        errors = []
        time_list = []
        predict_time = []
        train_time = []
        start_time = time.time()

        kf, inputs, outputs = self.format_data(data, k)
        
        for train, test in kf:
            try:
                t_time = time.time()
                objects = self.train_model(inputs, outputs, train)
                model = objects.pop()
                train_time.append(time.time() - t_time)

                t_time = time.time()
                predictions = self.make_predictions(model, inputs, test)
                predict_time.append(time.time() - t_time)

                errors.append(self.evaluate_error(outputs, test, predictions))
            finally:
                self.clean(objects)

        global_error = self.compute_global_error(outputs, errors)
        time_list = [time.time() - start_time, train_time, predict_time]
        return [[global_error, errors], time_list]

    ## Method which takes the raw data (lines of examples) and the number of folds k and returns kf, inputs
    # @param self the object pointer
    # @param data the raw_data (lines of examples, csv format) with the objective in the last comma separated field
    # @param k the number of folds for the k-fold cross validation
    # @return [kf,inputs,outputs] the train/test splits (see sklearn.cross_validation.Kfold doc for more info), inputs and outputs
    def format_data(self, data, k):
        
        data = shuffle(data, random_state=2014)

        # from the raw data we have in input, we create x (inputs) and y (outputs) arrays
        inputs = []
        outputs = []
        for elem in data:
            elem_list = elem.split(",")
            outputs.append(elem_list.pop())
            inputs.append(elem_list)
        inputs = np.array(inputs)
        outputs = np.array(outputs) 
        
        # these are arrays of integer for test/train positions
        kf = cross_validation.KFold(len(data), n_folds=k)

        self.is_regression_problem = (outputs[0][0] != "\"")

        return [kf, inputs, outputs]
    
    ## Abstract method which is used to train a regression or classification model
    # @param self the object pointer
    # @param inputs the inputs
    # @param outputs the outputs
    # @param train the integer array of positions for the data used for training
    # @return a list of objects related with the model, the last one has to be the one to interact with the trained model
    def train_model(self, inputs, outputs, train):
        raise NotImplementedError
    
    ## Abstract method which is use to make predictions using a model
    # @param self the object pointer
    # @param model an object used to interact with the trained model
    # @param inputs the inputs
    # @param test the integer array of positions for the data used for testing
    # @return a list of predictions for the test outputs given the test inputs
    def make_predictions(self, model, inputs, test):
        raise NotImplementedError
    
    ## Method which returns the error for a given list of predictions and a given list of outputs
    # @param self the object pointer
    # @param outputs the numpy array of outputs
    # @param test the numpy arrays of positions for test data
    # @param predictions the list containing the predictions
    # @return the error for the classification or for the regression
    def evaluate_error(self, outputs, test, predictions):
        if self.is_regression_problem:
            err = 0
            for i in range(0,len(outputs[test])):
                err += pow(predictions[i]-float(outputs[test][i]),2)
        else:
            # Error for the current model if it is classification
            err = 0.0
            for i in range(0, len(outputs[test])):
                if "\""+predictions[i]+"\"" != outputs[test][i]:
                    err += 1
        return err

    ## Abstract method to clean what has been created 
    # @param self the object pointer
    # @param objects the objects needed to clean
    def clean(self, objects):
        raise NotImplementedError

    ## Method to compute the mean squared error
    # @param self the object pointer
    # @param errors the error vector
    # @params outputs the outputs
    # @return the mean squared error
    def global_mean_squared_error(self, errors, outputs):
        return np.sqrt(np.sum(errors)/len(outputs))
    
    ## Method to compute the accuracy
    # @param self the object pointer
    # @param errors the error vector
    # @params outputs the outputs
    # @return the accuracy
    def accuracy(self, errors, outputs):
        return float(np.sum(errors))/len(outputs)

    ## Method used to return the global error (average of all the errors returned by evaluate_error())
    # @param self the object pointer
    # @param outputs the numpy array of outputs
    # @param errors the list containing the errors
    # @return the global error for the whole k-fold cross validation
    def compute_global_error(self, outputs, errors):
        if self.is_regression_problem:
            return self.global_mean_squared_error(errors, outputs)
        else:
            return self.accuracy(errors, outputs)

