import GenericKfold
from collections import Counter

## Class for doing a k-fold cross validation test over Predictions API
# It is a based on predictions with the average value of the outputs
class ModeKfold(GenericKfold.GenericKfold):
    
    ## Method which is use to train a regression or classification mode model
    # @param self the object pointer
    # @param inputs the inputs
    # @param outputs the outputs
    # @param train the integer array of positions for the data used for training # @review: what is this??
    # @return a dictionary with the key "prediction" containing value of the predictions (the average)
    def train_model(self, inputs, outputs, train):
        # @review: inputs are not used!
        # @review: only outputs[train] is used, why not passing this as arg instead of outputs and train?
        data = outputs[train]
        if self.is_regression_problem:
            import numpy as np
            average = float(sum(np.array(data, float)))/len(data)
            model = {"prediction": average}
        else:
            c = Counter(data)
            average_class, count = c.most_common()[0]
            model = {"prediction": average_class.split("\"")[1]}
        return [model]

    ## Abstract method which is use to make predictions using a mode model
    # @param self the object pointer
    # @param model a dictionary with key "prediction" and value the prediction made (average)
    # @param inputs doest not matter here
    # @param test the integer array of positions for the data used for testing
    # @return a list of predictions for the test outputs given the test inputs
    def make_predictions(self, model, inputs, test):
        predictions_list = [ model["prediction"] for k in range(0, len(test)) ]
        return predictions_list
    
    ## Method to clean what has been created: nothing here
    # @param self the object pointer
    # @param objects nothing
    def clean(self, objects):
        pass