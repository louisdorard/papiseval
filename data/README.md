# Folder contents

This directory contains the following data files:

  * houses.csv:      houses data (regression): number of beds/baths, surface, type, price
  * iris.csv:        iris flowers data (classification): petal/sepal length/width, species
	* kaggle-give-me-credit.csv: [Kaggle “give me credit” challenge](https://www.kaggle.com/c/GiveMeSomeCredit) (binary classification)
  * language.csv:    language data (classification): text, language
  * movielens.csv:   movie ratings data (regression): user id, movie id, rating
  * emails.csv:      emails data (classification): content features, sender-related features, importance
  * @todo new data files: cover, credit-application, diabetes


# Format

Each data file is a Comma Separated Values (CSV) file and it must have the following format:

 * first line is a header, e.g.: column one,column two,column three
 * each line corresponds to a data point / instance / example
 * all columns correspond to input features, except the last column which is the output
 * there are quotation marks around values when (and only when) they correspond to strings (i.e. they are categorical feature values or textual feature values)
 * each line must end with a new line character (\n), not a carriage return and a new line (\r \n)
 * the last line of the file is the last data point and must not be empty.

Also, for usage with this repository's code, there should be no missing values.

Note that having input features first then output in the last column is the format that BigML expects, but it differs from the Google Prediction API format where the output should be at the first column.