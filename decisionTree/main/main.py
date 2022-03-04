import pandas as pd
import math
import numpy as np

# degree of freedom table for vlaue 0-8 for confidence leve 0.99, 0.95 and 0.01
degree_freedom_table={'0.99':[0,0,0.02,0.115,0.297,0.554,0.872,0.989,1.344],
                      '0.95': [0,0.004,0.103,0.353,0.711,1.145,1.237,1.69,2.18],
                      '0.01':[0,10.828,13.816,16.266,18.467,20.515,22.515,24.322,26.124]}
user_input = input("press[E]for entropy,[G] for gini index,[M] for miscalculation error? ")
data_frame2=pd.DataFrame(degree_freedom_table)
# importing the csv files for train and test
data_train= pd.read_csv('train.csv');
data_test=pd.read_csv('test.csv')
df_train=pd.DataFrame(data_train)
df_test=pd.DataFrame(data_test)
# droping first column
df_train.drop(['1'],axis=1,inplace=True)

# spliting the giving file into the 1 to 60 attribute and droping column '1' and 'N'  assigning new column 'result'
count =0;
tot_array=[];
for x in range(0,60):
   tot_array.append([])
for y in range(0,60):
    count = 0;
    for x in df_train['N']:
        tot_array[y].append(df_train.loc[count, 'AAAAAAAAAATAGCTGGGCATGGTGGCAGGCGCCTGTAGTTTCAGCTGCTTGGTGTCTGA'][y]);
        count = count + 1;
counter=1;
df_train.drop('AAAAAAAAAATAGCTGGGCATGGTGGCAGGCGCCTGTAGTTTCAGCTGCTTGGTGTCTGA',axis=1,inplace=True);
for z in tot_array:
    df_train[counter]=z;
    counter= counter+1;
df_train['result']=df_train['N']
df_train.drop('N',axis=1,inplace=True)

features =list(df_train)
features.remove("result")
df_train.drop_duplicates(subset=features,inplace=True)

# same spliting of column 0-60 for test data and assiginign new column.
count =0;
tot_array=[];
for x in range(0,60):
   tot_array.append([])
for y in range(0,60):
    count = 0;
    for x in df_test['GCTGCACTGGATGGGACCTTCCAGAGGAAGGTAAGGCGTCTGATCCAGGTCTGGAGCTGG']:
        tot_array[y].append(df_test.loc[count, 'GCTGCACTGGATGGGACCTTCCAGAGGAAGGTAAGGCGTCTGATCCAGGTCTGGAGCTGG'][y]);
        count = count + 1;
counter=1;
df_test.drop('GCTGCACTGGATGGGACCTTCCAGAGGAAGGTAAGGCGTCTGATCCAGGTCTGGAGCTGG',axis=1,inplace=True);
for z in tot_array:
    df_test[counter]=z;
    counter= counter+1;

features1 = list(df_train)
df_train.drop_duplicates(subset=features, inplace=True)

# function the returns the 2d array with the uniwue vlaue in the column wiht its cound
# if you pass a dataframe and columnn name
def unique_sum(data_frame,colName):
    unique_value=data_frame[colName].unique()
    result=[]
    for val in unique_value:
        result.append([val,(data_frame[colName]==val).sum()])
    return result


# return multidimension array  of unique value(A,T,G,C,etc) count by their 'result column'(IE,EI,N)
# by taking dataframe and column name
def colm_unique_sum(data_frame,colName):
    result=[]
    for x in unique_sum(data_frame,colName):
        result.append([x[0],[
            ['EI', ((data_frame[colName] == x[0]) & (data_frame['result'] == 'EI')).sum()],
            ['IE', ((data_frame[colName] == x[0]) & (data_frame['result'] == 'IE')).sum()],
            ['N', ((data_frame[colName] == x[0]) & (data_frame['result'] == 'N')).sum()]
        ]])
    return result

#  the method return the  uniqur vlaue percent in 'result' column.
def chi_percent(data_frame):
    uni=unique_sum(data_frame,'result')
    res=[]
    total=len(data_frame)
    for x in uni:
        res.append([x[0],round((x[1]/total),3)])

    res.reverse()
    return res

# returns the chi square vlaue of the passed column name in data_frame
def chi_square_test(data_frame, colname):
    expec=expected_value(data_frame,colname)
    original= colm_unique_sum(data_frame,colname)
    chi_square= 0;
    counter=0;
    for k in expec:
        counter2=0
        for inner in k[1]:
            var=(original[counter][1][counter2][1]-inner[1]) **2
            if(inner[1]!=0):
                chi_square+=var/ inner[1]
            counter2+=1
        counter+=1;
    return chi_square

# return the degree of freedom for the dataframe given
def degree_of_freedom(data_frame, column):
    col_length= len(unique_sum(data_frame,column))
    row_length= len(unique_sum(data_frame,'result'))
    return((col_length-1)* (row_length-1))

# return true or false by comparing the chi_sqaure_test function with the degree of freedom table
#true is chi square test value is > either of '0.99','0.95','0.01' value from table otherwise return false.
def check_chi_square_test(data_frame,colnum):
    col= degree_of_freedom(data_frame,colnum)
    if(chi_square_test(data_frame,colnum)>data_frame2.loc[col]['0.99']
            or chi_square_test(data_frame,colnum)>data_frame2.loc[col]['0.95']
            or chi_square_test(data_frame,colnum)>data_frame2.loc[col]['0.01']):
        return True
    else:
        return False

# Calculates the impurity in a node
def mis_calc_imp():
    tot_val_col = []
    for x in unique_sum(df_train, "result"):
        tot_val_col.append(x[-1])
    total_error = 1 - (max(tot_val_col)) / (sum(tot_val_col))
    return round(total_error, 3)


#Calculates miss-classification error of each attribute
def misc_calc(data_frame, colName):
    arr = max(colm_unique_sum(data_frame, colName))
    sum_tots = len(data_frame)
    for x in arr:
        get_max_attribute = max(x)[-1]
    max_error_each_attribute = 1 - (get_max_attribute / sum_tots)
    return round(max_error_each_attribute, 3)

#Calculates information gain using miss-classification error
def max_information_gain_miscal(data_frame):
    col = list(data_frame.columns)
    col.remove('result')
    resut_col = col[0]
    max = misc_calc(data_frame, col[0])
    for c in col:
        if (max < misc_calc(data_frame, c)):
            max = misc_calc(data_frame, c)
            resut_col = c
    return resut_col


def expected_value(data_frame,columName):
    per=chi_percent(data_frame)
    uni_colm=colm_unique_sum(data_frame,columName)
    uni_sum= unique_sum(data_frame,columName)
    count1=0
    for a in uni_colm:
        count2 = 0
        for b in a[1]:
            try:
                b[1]=round(per[count2][1]*uni_sum[count1][1])
            except:
                b[1]=1
            count2+=1
        count1=0
    return uni_colm

# return gini_index vlaue of the given array i.e allarray is the 2d array format for
# form like unique_sum function return.
def gini_index(allarray):
    sample = 0;
    gini = 1;
    for x in allarray:
        sample = sample + x[1]
    for y in allarray:
        gini -= (y[1] / sample) ** (2)

    return round(gini,3)
#return the information giain of the column number provided
# it uses colm_unique_sum,gini_index function.
def information_gain_gini(data_frame,colname):
    arr= colm_unique_sum(data_frame,colname)
    total_sample= len(data_frame)
    expectec_gini=0;
    c=0
    for z in arr:
        cont=0
        for x in z[1]:
            cont+=x[1]
        cont=cont/total_sample
        expectec_gini+=cont*gini_index(z[1])
        c+=1
    return (round(expectec_gini,3))

# return the 'columnname' of minimum gini index
# it uses information_gain_ginin function.
def max_information_gain_gini(data_frame):
    col = list(data_frame.columns)
    col.remove('result')
    resut_col = col[0];
    max = information_gain_gini(data_frame, col[0])
    for c in col:
        if (max > information_gain_gini(data_frame, c)):
            max = information_gain_gini(data_frame, c)
            resut_col = c;
    return resut_col
# return entropy vlaue of the given array i.e allarray is the 2d array format for
# form like unique_sum function return
def entropy_func(allarray):
    sample=0;
    entropy =0;
    for x in allarray:
        sample= sample+x[1]
    for y in allarray:
        val1 = y[1] / sample
        try:\
            entropy += (val1 * math.log(val1, 2.0))
        except:
            k=0
    return (-1*entropy)
#return the information giain of the column number provided using entropy
# it uses colm_unique_sum,entropy_func function.

def information_gain_entropy(data_frame,colname):
    arr= colm_unique_sum(data_frame,colname)
    total_sample=len(data_frame)
    tot_entropy=entropy_func(unique_sum(data_frame,'result'))
    expected_entropy=0;
    c=0
    for z in arr:
        cont=0
        for x in z[1]:
            cont+=x[1]
        cont=cont/total_sample
        expected_entropy+=cont*entropy_func(z[1])
        c+=1
    return (round(tot_entropy-expected_entropy,3))

# return the 'columnname' of maximum information gain with entropy
# it uses information_gain_entropy function.
def max_information_gain_entropy(data_frame):
    col=list(data_frame.columns)
    col.remove('result')
    resut_col=col[0];
    max= information_gain_entropy(data_frame,col[0])
    for c in col:
        if ( max< information_gain_entropy(data_frame,c)):
            max=information_gain_entropy(data_frame,c)
            resut_col=c;
    return resut_col
# create a class node
#initialize the childern array , value string, isLeaf boolean and leafValue string
# childeren to add the new node, value to know attribute , isleaf to check the leaf node or not, leaf_value is the predected pure leaf ['IE','EI,'N']
class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.leafValue = ""

#it creats the resulting tree.
def Tree(data_frame):
    #list of attribute in data frame
    attr=list(data_frame.columns)
    # removing the resulting colum form attr
    attr.remove('result')
    # intitalizing the root node
    root = Node()
    # checking the user input to build the tree form entropy,gini index or misclaculation error
    if (user_input == 'E'):
        max_feat = max_information_gain_entropy(data_frame)
    elif (user_input == 'G'):
        max_feat = max_information_gain_gini(data_frame)
    elif (user_input == 'M'):

        max_feat = max_information_gain_miscal(data_frame)
    else:
        #gini index as defualt parameter
        print("you give the worg input so the tree will be build on deafault arugument[G] gini index.")
        max_feat = max_information_gain_gini(data_frame)
    # root node become max_feat(max value of atribute)
    root.value = max_feat
    # unique colum in dataframe
    uniq = np.unique(data_frame[max_feat])
    for u in uniq:
        #creating a sub data.
        subdata = data_frame[data_frame[max_feat]==u]
        del subdata[max_feat]
        #unique value in result colum of subdata
        uniqe = np.unique(subdata["result"])
        check_test=check_chi_square_test(data_frame,max_feat)
        # checking if it is leaf node
        if (uniqe.size==1):
            newNode = Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.leafValue = np.unique(subdata["result"])[0]
            root.children.append(newNode)
        # checking forom the chi square test if the split not good then  makeing it leaf node with value IE
        elif(check_test==False):
            newNode = Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.leafValue = 'IE'
            root.children.append(newNode)
        else:
        # else creatign a new node to add on the root node.
            tempNode = Node()
            tempNode.value = u
            child = Tree(subdata)
            tempNode.children.append(child)
            root.children.append(tempNode)
    return root
# fuction to search through the tree formed
def searchNode(node, data, row):
    root = node
    result=""
    for child in root.children:
        if (child.value==data.loc[row][root.value]):
            if (child.isLeaf):
                result = child.leafValue
            else:
                result = searchNode(child.children[0], data, row)
    return result
# return the aray with the predicted result.
def predict(tree, test_d):
    ans = []
    for i in range(len(test_d)):
        ans.append(searchNode(tree, test_d, i))
    return ans

# print the tree to see it.
def printTree(root: Node, height=0):
    for i in range(height):

        print("\t", end="")
    print(root.value, end="")
    if root.isLeaf:
        print(" == ", root.leafValue)
    print()
    for child in root.children:
        printTree(child, height + 1)

root = Tree(df_train)
result = predict(root, df_test)
df_test['id']=df_test['2001']
df_test['class']=result
feature3=list(df_test.columns)
feature3.remove('id')
feature3.remove('class')
for x in feature3:
    df_test.drop([x],inplace=True,axis=1)
# uncomment to export the output to csv file with your local file name
df_test.to_csv(r"C:\Users\15635\PycharmProjects\u_test_result_msc.csv",index=False)
printTree(root)
print(df_test)