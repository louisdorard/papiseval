from bigml.api import BigML
from bigml.model import Model
import numpy as np
import os
from sklearn.base import BaseEstimator, RegressorMixin

class simple_model(BaseEstimator, RegressorMixin):
    """Creates a simple model (decision tree) on BigML"""

    def __init__(self):

        self.api_ = BigML(dev_mode=True)
        if (len(self.api_.list_projects("name=scikit")['objects']) != 0):
            self.api_.delete_project(self.api_.list_projects('name=scikit')['objects'][0]['resource'])
        self.project_ = self.api_.create_project({'name': 'scikit'})


    def fit(self, X, y):

        api = self.api_
        project = self.project_

        # concatenate X and y
        data = np.concatenate((X, np.array([y]).T), axis=1).tolist()

        # Create training data source (this is mostly about uploading the file so we won't time it)
        source = api.create_source(data, {"project": project["resource"], "name": "training"})
        api.ok(source)

        # Train model from dataset created from this source
        # @todo: time this block and save it as a property of the object (which can later be used by skll when reporting results)
        dataset = api.create_dataset(source, {"name": "training"})
        api.ok(dataset)
        self.model_ = api.create_model(dataset)
        api.ok(self.model_)

        # Download offline model
        # self.local_model_ = Model(self.model_)

        return self

    def predict(self, Xtest):

        api = self.api_
        project = self.project_

        # Create test data source (this is mostly about uploading the file so we won't time it)
        source = api.create_source(Xtest.tolist(), {"project": project["resource"], "name": "test"})
        api.ok(source)

        # @todo: time this block and save it as a property of the object (which can later be used by skll when reporting results)
        dataset = api.create_dataset(source, {"name": "test"})
        api.ok(dataset)
        batch_prediction = api.create_batch_prediction(self.model_, dataset, {
                "all_fields": True, "header": True, "confidence": True})
        api.ok(batch_prediction)

        # Get back the predictions (this is mostly about downloading the predictions so we won't time it)
        api.download_batch_prediction(batch_prediction, filename='./my_predictions.csv')
        ytest = np.genfromtxt('./my_predictions.csv', delimiter=',')[1:,-2]
        os.remove('./my_predictions.csv')

        # Use offline model
        # ytest = self.local_model_.predict(X)

        return ytest
