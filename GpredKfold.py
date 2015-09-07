import GenericKfold
import StringIO
import os
import shutil
import tempfile
import time
import threading
import numpy as np
import boto
import gcs_oauth2_boto_plugin
import os
from apiclient import http

## Class for doing a k-fold cross validation test over Google Predictions API
class GpredKfold(GenericKfold.GenericKfold):
    def __init__(self, api):
        self.api = api
        self.project_id = os.environ['GPRED_PROJECT_ID']
        # URI scheme for Google Cloud Storage.
        self.GOOGLE_STORAGE = 'gs'
        self.counter = 0 #debug

    ## Method to reinitialize the api (after a long wait the access expires)
    # @param the pointer object
    def reinitialize_api(self):
        import googleapiclient.gpred as gpred
        self.api = gpred.api(os.environ['GPRED_OAUTH_FILE'])

    # upload_file is used to upload file on the google cloud storage
    #
    # params:
    #           dir          the directory where the file is
    #           filename     the name of the file you want to upload
    #           bucket_name  the name of the bucket which will be used
    def upload_file(self, dir, filename, bucket_name):

        with open(os.path.join(dir, filename), 'r') as localfile:

            dst_uri = boto.storage_uri(bucket_name + '/' + filename, self.GOOGLE_STORAGE)
            dst_uri.new_key().set_contents_from_file(localfile)

    # define_bucket: setting a bucket which will be used to store training data
    def create_bucket(self):
        now = time.time()
        BUCKET = 'training-%d' % now

        # Instantiate a BucketStorageUri object.
        uri = boto.storage_uri(BUCKET, self.GOOGLE_STORAGE)
        # Try to create the bucket.
        try:
        # If the default project is defined,
        # you do not need the headers.
        # Just call: uri.create_bucket()
            header_values = {"x-goog-api-version": "2",
            "x-goog-project-id": self.project_id}
            uri.create_bucket(headers=header_values)
            return BUCKET
        except boto.exception.StorageCreateError, e:
            print 'Failed to create bucket:', e

    # delete_bucket: deleting a bucket
    #
    # params:
    #           bucket_name: the name of the bucket to delete
    #           project_id: the project id where the bucket to delete is
    def delete_bucket(self, bucket_name, project_id):

        uri = boto.storage_uri(bucket_name, self.GOOGLE_STORAGE)
        for obj in uri.get_bucket():
            #print 'Deleting object: %s...' % obj.name
            obj.delete()
        #print 'Deleting bucket: %s...' % uri.bucket_name
        uri.delete_bucket()

## Method which is use to train a google prediction regression or classification model
    # @param self the object pointer
    # @param inputs the inputs
    # @param outputs the outputs
    # @param train the integer array of positions for the data used for training
    # @return a list containing the bucket name (to clean it at the end) and the model id
    def train_model(self, inputs, outputs, train):
        # Create a file with the trained data
        filename = "data_train.csv"
        f = open("./data_train.csv", "w")

        for x0, y0 in zip(inputs[train],outputs[train]):
            y0 = np.array(y0)
            line = ",".join(np.insert(x0, 0, y0))
            f.write(line+"\n")
        f.close()

        #We create a bucket and keep its name
        bucket_name = self.create_bucket()

        self.upload_file('./',filename, bucket_name)

        id = "datatrain"
        body = {
                "id": id,
                "storageDataLocation": bucket_name+"/"+filename
                }

        self.api.trainedmodels().insert(project = self.project_id,
                                   body = body).execute()

        get_request = self.api.trainedmodels().get(project = self.project_id,
                                              id = id)

        # Wait for the training to complete.
        while True:
            status = self.api.trainedmodels().get(project = self.project_id, id = id).execute()
            state = status['trainingStatus']
            print 'Training state: ' + state
            if state == 'DONE':
                break
            elif state == 'RUNNING':
                time.sleep(10)
                continue
            else:
                raise Exception('Training Error: ' + state)

        # Job has completed.
        print 'Training completed'

        return [bucket_name, id]

    ## Method which is use to make predictions using a google prediction model
    # @param self the object pointer
    # @param model an object used to interact with the trained model
    # @param inputs the inputs
    # @param test the integer array of positions for the data used for testing
    # @return a list of predictions for the test outputs given the test inputs
    def make_predictions(self, model, inputs, test):

        predictions_list = []

        # Loop over the inputs in the test set to make predictions based on them
        for x0 in inputs[test]:
            self.counter += 1 #debug
            print self.counter #debug

            # This list and dictionary will be used as predictions inputs
            input_data = {}
            data_list = []
            for i in range(0, len(x0)):
                data_list.append(x0[i])
            # We build the input data for our model
            input_data = {
                          "input": {
                                    "csvInstance": data_list
                                    }
                          }
            try:
                # Make prediction
                res = self.api.trainedmodels().predict(project = self.project_id,
                                                        id = model,
                                                        body = input_data).execute()
            except http.HttpError:

                print 'wait 240s'
                time.sleep(240)
                self.reinitialize_api()
                 # Make prediction
                res = self.api.trainedmodels().predict(project = self.project_id,
                                                        id = model,
                                                        body = input_data).execute()

            if self.is_regression_problem == True:
                prediction = res['outputValue']
                predictions_list.append(float(prediction))
            else:
                prediction = res['outputLabel']
                predictions_list.append(prediction)

        return predictions_list

    ## Method to clean what has been created
    # @param self the object pointer
    # @param objects the objects needed to clean: model and bucket (google prediction objects)
    def clean(self, objects):
        self.api.trainedmodels().delete(project = self.project_id,
                                   id = "datatrain").execute()
        self.delete_bucket(objects.pop(), self.project_id)
        os.remove("./data_train.csv")
