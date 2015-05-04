# papiseval

Evaluate Predictive APIs

## How to launch an evaluation

The evaluation runs cross-validation procedures on a dataset specified by you.

The command to launch is:

> python evaluate.py --filename=yourpath/yourfilename.csv --input=yourinputformat --k=yournumberoffolds --services=bigml,gpred,mode

The output is the performance measures (both error and time) for all methods ([Google Prediction API](http://cloud.google.com/prediction/), [BigML](http://www.bigml.com/), baseline).


# Example

Launch:

> python evaluate.py --filename=data/language-detection.csv --k=2 --services=mode

The output would be:

>	Mode 2-fold:
		  Error :      0.614814814815
		  R squared :  0
		  Time  :      0.00482296943665

	More details :
		  Errors fold by fold:
		124.0
		125.0
		Training times fold by fold:
		0.000163793563843
		0.0
		Prediction times fold by fold:
		1.28746032715e-05
		0.0
	[...]

At first it is recommended to only include "mode" as a service, as a check that the procedure runs correctly through the data you specified. You can try first with a small number of folds (2), with "bigml" (usually quicker) and then with "gpred". Then, it is standard practice to use k=10 folds for a proper evaluation.

# Architecture of the evaluation code

## evaluate.py

This is the main script. It uses the bigml kfold cross-validation and gpred cross-validation files in order to estimate the error and the time taken by the predictions API.

When you launch it, it creates a log.txt file with the results, or it appends the results to log.txt if it already exists. The results are printed in the console as well.

## utils.py

This file contains everything needed to read data. In particular, the read_data() method reads a .csv file and returns an array which is used in the kfoldcv (k-fold cross validation) methods.

## gpred.py

This file contains a method called Gpred() which is used to create a Google Prediction API object ('api') to interact with the API (sending requests and receiving responses). You have to feed the api as a parameter to the *_kfold methods (discussed below).

## generic_kfold.py

This file contains an abstract class, "Generic_Kfold", that implements all of the common methods for the k-fold cross-validation procedure. It also specifies the functions that need to be implemented in the concrete API classes (BigML_Kfold, Gpred_Kfold and Mode_Kfold). 

## mode_kfold.py

This file contains a concrete class, "Mode_Kfold", that implements the abstract class "Generic_Kfold". It cross-validates the mode predictor, which is then useful to serve as a baseline to compare actual predictors to.

## gpred_kfold.py

This file contains a concrete class, "Gpred_Kfold", that implements the abstract class "Generic_Kfold". It cross-validates the Gpred-based predictor through the kfoldcv() method of the parent class (which performs a kfold cross-validation on the list returned by read_data()).

## bigml_kfold.py

This file contains a concrete class, "Bigml_Kfold", that implements the abstract class "Generic_Kfold". It cross-validates the BigML-based predictor through the kfoldcv() method of the parent class.
