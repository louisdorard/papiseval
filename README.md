# papiseval

Evaluate Predictive APIs.

This repository contains Python code to evaluate the performance (time and error) of BigML and Google Prediction API (Gpred) on a given dataset.


## How to run evaluations

Evaluations are run as experiments in [SKLL (SciKit-Learn Laboratory)](https://github.com/EducationalTestingService/skll).

### Using Docker

Building the image:

> docker build -t="louisdorard/papiseval” .

Running a docker container from this image:

> docker run -d -v $PWD:/home/jovyan/work --name papiseval louisdorard/papiseval

One the container is running we can use it to execute experiments:

> docker exec papiseval run_experiment -l boston/evaluate.cfg

The summary of results can be accessed at output/Boston_Evaluate_summary.tsv


## Contents of this repo

* Dockerfile
* mode.py: mode scikit classifier
* bigml_scikit.py: scikit regressor that uses BigML
* Scikit.ipynb: notebook demonstrating the above
* 1 directory per datafile containing
	* code to download & format data
	* experiment config files (e.g. evaluate, cross validate, ...)
	* train/ : directory containing training data files
	* test/ : directory containing test data files
* boston/ : taken from the examples in SKLL 1.0.0
* output/ : where the outputs from the experiments will be
