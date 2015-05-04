import numpy as np

# @todo: document inputs and outputs

# format_data: gives an array with all the datas formatted and another with the output only
#
# params:
#			filename         the name of the file to read
#			objective        the column number of the objective starting from 0
#			new_objective    the new column number where we want to store the objective
#			line_delimiter   the symbol which separates the inputs
#			field_delimiter  the symbol which separates the input features
#
# returns:
#			x: a numpy array containing all the datas
#			y: a numpy array containing only the objectives
def format_data(filename, objective, new_objective, line_delimiter="\n", field_delimiter=","):
	f = open(filename)
	content = f.readlines()
	res = []
	y = []
	temp = []
	lines = "".join(content).split(line_delimiter) #to remove the line delimiter
	lines.pop() #to remove the header
	x = lines
	for obj in x:
		temp = obj.split(field_delimiter)
		y.append(temp[objective])
		obj = temp.pop(objective) # @todo enlever pour avoir seulement le resultat
		temp.insert(new_objective, obj) # @todo enlever pour avoir seulement le resultat
		res.append(field_delimiter.join(temp))
	x = res
	x = np.array(x)
	y = np.array(y)
	return [x,y]

# @todo: document inputs and outputs
# read_data
#
# params:
#			filename  name of the .csv file we want to read
#					  it has to be in a predefined shape:
#					  to separates elements: "\n"
#					  to separates fields of one element: ","
#					  the objective field has to be the last field of one element
#
# returns:
#			data  a numpy array containg the datas as strings
#				  one element of this data is an input and the related objective
def read_data(filename):
    f = open(filename)
    content = f.readlines()
    #removing the header
    content.pop(0)
    #this is our input
    res = content
    #this is our output
    data = []
    for elem in res:
        data.append((elem.split("\n")[0]))
    data = np.array(data)
    return data

# used to rewrite_data, not detailed yet (because yet not often used)
def rewrite_data(filename, new_filename, objective, new_objective, line_delimiter="\n", field_delimiter=","):
	f = open(filename)
	content = f.readlines()
	res = []
	y = []
	temp = []
	lines = "".join(content).split(line_delimiter) #to remove the line delimiter
	lines.pop()
	x = lines
	for obj in x:
		temp = obj.split(field_delimiter)
		y.append(temp[objective])
		obj = temp.pop(objective) # @todo enlever pour avoir seulement le resultat
		temp.insert(new_objective, obj) # @todo enlever pour avoir seulement le resultat
		res.append(field_delimiter.join(temp))
	f.close()
	f2 = open(new_filename, "w")
	for elem in res:
		f2.write(elem+"\n")
	f2.close()