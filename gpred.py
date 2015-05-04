import argparse
import httplib2
import os
import sys

from apiclient import discovery
from oauth2client import file
from oauth2client import client
from oauth2client import tools

# Gpred: authenticates into the Google Prediction API and returns an object to use to issue requests to the API.
#
# This code comes from sample.py in the starter application provided by https://developers.google.com/api-client-library/python/ when selecting the Prediction API in Command Line at the bottom of that page. It has been modified to fit our needs.

def Gpred():

  # If the credentials don't exist or are invalid run through the native client
  # flow. The Storage object will ensure that if successful the good
  # credentials will get written back to the file.
  storage = file.Storage('/vagrant/credentials/gpred_credentials.dat')
  credentials = storage.get()
  if credentials is None or credentials.invalid:
    print("Please launch Gpred_API_credentials_setup.sh because you do not have"
          +"gpred_credentials.dat\n")
    return

  # Create an httplib2.Http object to handle our HTTP requests and authorize it
  # with our good Credentials.
  http = httplib2.Http()
  http = credentials.authorize(http)

  # Construct the service object for the interacting with the Prediction API.
  api = discovery.build('prediction', 'v1.6', http=http)

  try:
      return api

  except client.AccessTokenRefreshError:
      print ("The credentials have been revoked or expired, please re-run"
        "the application to re-authorize")