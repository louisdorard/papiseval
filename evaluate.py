#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import utils, sys

## Function to print the help of the main
def helpmain():
    """ Function to print the help of the main """
    print("""
        Evaluation script

        Example:
            python evaluate.py --filename=/vagrant/data/language.csv --k=2 --services=mode

        Description of the required arguments:
            --filename :  the path where your .csv file containing your data is stored.
            --k :         the number of folds you want for the k-fold cross-validation.
            --services :  the services you want to use (gpred, bigml, mode).
        
        For further information, you can check the README.md file in this folder.
        """)

## Function to print the results of k-fold cross validations
# @param api_name the name of the api we are printing the results for
# @param res the errors list returned by the kfoldcv of the api
# @param time the time vector returned by the kfoldcv of the api
# @param mode_res the errors list returned by the Mode kfoldcv
def print_results(api_name, res, time, mode_res):
    """ Function to print the results of k-fold cross validations """

    print "\n"+api_name+" "+str(k)+"-fold:"  
    print "  Error :      {}".format(res[0])
    # if mode_res == 0 then we know we want to print res for Mode_Kfold
    if mode_res == 0:
        print "  R squared :  {}".format(0)
    else:
        print "  R squared :  {}".format(1-pow(res[0]/mode_res[0],2))
    print "  Time  :      {}".format(time[0])
    print "\nMore details :"
    print "  Errors fold by fold: "
    for e in res[1]:
        print "    {}".format(e)
    print "  Training times fold by fold: "
    for t in time[1]:
        print "    {}".format(t)
    print "  Prediction times fold by fold: "
    for t in time[2]:
        print "    {}".format(t)

# default values
filename = "/vagrant/data/language.csv"
services = "mode"
k = 2

# we set up the variables we will need from the command line
sys.argv.pop(0)
if sys.argv == []:
    helpmain()
    sys.exit()

for arg in sys.argv:
    arg_type = arg.split("=", 1)[0]
    
    if arg == "--help":
        helpmain()
        sys.exit()

    if arg.find("=") != -1:
        arg_value = arg.split("=", 1)[1]
    else:
        print "\n** Error ! Unrecognized argument ! Let me print the help **"
        print "\n  ----------------------------------------------------------\n"
        helpmain()
        sys.exit(-1)

    if arg_type == "--filename":
        filename = arg_value
    elif arg_type == "--k":
        k = int(arg_value)
    elif arg_type == "--services":
        temp = arg_value.split(",")
        services = temp
    else:
        print "\n** Error ! Unrecognized argument ! Let me print the help **"
        print "\n  ----------------------------------------------------------\n"
        helpmain()
        sys.exit(-1)

#                   #
#       Mode        #
#                   #

import ModeKfold

# we read the datas
data = utils.read_data(filename)

obj0 = ModeKfold.ModeKfold()

# we do a mode k-fold cross-validation
res0, time0 = obj0.cross_validation(data,k, "")

if "mode" in services:
    # we print the results
    print_results("Mode", res0, time0, 0)

#                   #
#       BigML       #
#                   #

if "bigml" in services:
    
    # @review: only import if and where needed
    import BigmlKfold
    from bigml.api import BigML

    # we set up the bigml api
    api = BigML(dev_mode=True)

    obj1 = BigmlKfold.BigmlKfold(api)

    # we launch the k-fold cross validation and gather the error and the time taken
    res1, time1 = obj1.cross_validation(data, k, api)

    # we print the results
    print_results("BigML", res1, time1, res0)

#                   #
#       Gpred       #
#                   #

if "gpred" in services:
    
    import GpredKfold
    from gpred import Gpred

    import os
    googleplus_project_id = os.environ['GPRED_PROJECT_ID']

    # we set up the gpred api
    api = Gpred()

    obj2 = GpredKfold.GpredKfold(api)

    # we launch the k-fold cross validation and gather the error and the time taken
    res2, time2 = obj2.cross_validation(data, k, api)

    # we print the results
    print_results("Gpred", res2, time2, res0)

#                   #
#        Log        #
#                   #

# we store the results in a log.txt file if we want to compare
# f = open("log.txt", "a")

# f.write("\nFor the data in the file : "+filename+"\n"
#         +"We proceeded a "+str(k)+"-fold cross-validation, below are the results:\n\n"
#         +"Results for **Mode**\n"
#         +"    Error/Accuracy : "+str(res0[0])+"\n"
#         +"    Time: "+str(time0)+"\n"
#         +"    R squared: 0\n\n"
#         +"Results for **BigML**\n"
#         +"    Error/Accuracy : "+str(res1[0])+"\n"
#         +"    Time: "+str(time1)+"\n"
#         +"    R squared: "+str(1-pow(res1[0]/res0[0],2))+"\n\n"
#         +"Results for **Gpred**\n"
#         +"    Error/Accuracy : "+str(res2[0])+"\n"
#         +"    Time: "+str(time2)+"\n"
#         +"    R squared: "+str(1-pow(res2[0]/res0[0],2))+"\n\n"
#         +" -----------------\n")
# f.close()