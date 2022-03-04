#Decision Tree. 

Decision Tree is simply defined as a tree representation used for making decision and widely used alogrithm for machine 
learning. In this project, decision tree is implemented for training DNA dataset and later calculating accuracy with the
help of testing dataset.

In this project, decision tree algorithim is created from scratch using ID3 implementation. During the formation of 
each node, Chi sqaure test is carried out checking the randomness of the data. Also error calculations like gini index,
entropy and miss-classification error are carried out. Each error test can be used to form different trees.
i.e you can form a decision tree with gini index or with entropy or with miss-classification error. Accuracies for split
stopping at 99, 95, 0 confidence are also used. 

## To run the program: 
    The program is written in python language. When you run the program it asks the user in command line argument with 
the following command.

" press[E]for entropy,[G] for gini index,[M] for miscalculation error? "
The user has to either enter E, G or M depending upon the tree they want to create with the following error calculation.
If the user enters wrong command then the tree is created using default mode which is with entropy

**##Method:** 

**unique_sum ():**
    This function returns two dimentional array count of unique value present in coloumn
    This function takes dataframe and column name as a parameter.

**colm_unique_sum():**
    This function returns multidimensional array count of unique value which is (A,T,G,C ..) with respect to the result
    column which consist of (N, IE, EI)
    This function takes dataframe and column name as parameter.

**chi_percent():**
    This method returns the unique value percent in result column

**chi_square_test()**
    This method measures the randomness of value distributed in an attribute.
    This method returns chi square value of the passed column in the dataframe.
    Returns chi_square value

**degree_of_freedom()**
    This method calculates the degree of freedom of a given dataframe

**check_chi_square_test()**
    This method compares chi square test with degree of freedom and returns boolean value
    returns true/false depending upon the test

**mis_calc_imp():**
    This method measures the impurity present in the node

**mis_cal():**
    This method calculates the miss-classification value

**max_information_gain_miscal():**
    This method calculates the information gain using miss-classification error.

**gini_index():**
    This method returns the value after calculation of gini index

**information_gain_gini():**
    This method calculates the information gain using gini index.

**max_information_gain_gini():**
    Returns the max information gain found using gini index and then is used for the upcoming node.

**entropy_func():**
    This funciton calculates the entropy value of a given array

**information_gain_entropy():**
    This function calculates the information gain using entropy

**max_information_gain_entropy():**
    Returns the max information gain found using entropy and then is used for the upcoming node.


##Class Node:
    This is a class, which initializes array, creates value string, isLeaf boolean and leafValue.
    children: This addes new node to the tree
    value: this add the attribute to the node
    isLeaf: checks if the node is a leaf node or not
    leafValue: is the predicted leaf

###Method:
**Tree():**
    Main method which creates the decision tree using ID3 implementation

**searchNode():**
    After formation of the tree, this method is used to search within the tree

**predict():**
    Returns an array of predicted result.

**printTree():**
    Displays the tree formed in the command line terminal


Thus, these are the methods and implementation of the project. Also at the very end output file is exported to a csv 
file as stated in the requirements folder.

