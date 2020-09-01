from mouselight_code.src import read_swc
import numpy as np
from pathlib import Path
import pandas as pd
import re
from random import shuffle
import math
import matplotlib.pyplot as plt

def read_swc_basic(path):
    """Read a single swc file

    Arguments:
        path {string} -- path to file

    Returns:
        df {pandas dataframe} -- indices, coordinates, and parents of each node
    """
    # read coordinates
    df = pd.read_table(
        path,
        sep = ",",
        delimiter= None,
        names=["sample", "structure", "x", "y", "z", "r", "parent"],
        delim_whitespace=True,
    )
    return df

total_correct = 0
total_incorrect = 0
total_length = 0

#create matrix to save (1) num correct (2) num incorrect (3) percent correct (4) percent incorrect (5) length
data_matrix = np.zeros((5,100))

for p in range(1,101):

    print("round " + str(p))

    #old swc files
    swc_file_old_info = open("C:\\Users\\chsha\\Downloads\\gcut\\gcut\\matlab\\demo_data\\imp_info\\imp_info_" + str(p) + ".txt") 
    swc_file_old_1_path = swc_file_old_info.readline().rstrip()
    swc_file_old_2_path = swc_file_old_info.readline().rstrip()
    swc_file_old_1 = read_swc.read_swc(swc_file_old_1_path)
    swc_file_old_2 = read_swc.read_swc(swc_file_old_2_path)
    swc_file_old_1 = swc_file_old_1[0]
    swc_file_old_2 = swc_file_old_2[0]

    #new swc files 
    swc_file_new_1_path = "C:\\Users\\chsha\\Downloads\\gcut\\gcut\\matlab\\demo_result\\mat_file_" + str(p) + "_split_1.swc"
    swc_file_new_2_path = "C:\\Users\\chsha\\Downloads\\gcut\\gcut\\matlab\\demo_result\\mat_file_" + str(p) + "_split_2.swc"
    swc_file_new_1 = read_swc_basic(swc_file_new_1_path)
    swc_file_new_2 = read_swc_basic(swc_file_new_2_path)

    #save x y z location of swc files (Truncated and rounded to cover all)
    old_1_trunc = np.zeros((len(swc_file_old_1),3))
    old_2_trunc = np.zeros((len(swc_file_old_2),3))
    new_1_trunc = np.zeros((len(swc_file_new_1),3))
    new_2_trunc = np.zeros((len(swc_file_new_2),3))
    old_1_round = np.zeros((len(swc_file_old_1),3))
    old_2_round = np.zeros((len(swc_file_old_2),3))
    new_1_round = np.zeros((len(swc_file_new_1),3))
    new_2_round = np.zeros((len(swc_file_new_2),3))

    #save truncated and rounded values for each swc file 
    for index in swc_file_old_1.index:
        old_1_trunc[index][0] = math.trunc(swc_file_old_1.loc[index,'x'])
        old_1_trunc[index][1] = math.trunc(swc_file_old_1.loc[index,'y'])
        old_1_trunc[index][2] = math.trunc(swc_file_old_1.loc[index,'z'])
        old_1_round[index][0] = round(swc_file_old_1.loc[index,'x'])
        old_1_round[index][1] = round(swc_file_old_1.loc[index,'y'])
        old_1_round[index][2] = round(swc_file_old_1.loc[index,'z'])

    for index in swc_file_old_2.index:
        old_2_trunc[index][0] = math.trunc(swc_file_old_2.loc[index,'x'])
        old_2_trunc[index][1] = math.trunc(swc_file_old_2.loc[index,'y'])
        old_2_trunc[index][2] = math.trunc(swc_file_old_2.loc[index,'z'])
        old_2_round[index][0] = round(swc_file_old_2.loc[index,'x'])
        old_2_round[index][1] = round(swc_file_old_2.loc[index,'y'])
        old_2_round[index][2] = round(swc_file_old_2.loc[index,'z'])

    for index in swc_file_new_1.index:
        new_1_trunc[index][0] = math.trunc(swc_file_new_1.loc[index,'x'])
        new_1_trunc[index][1] = math.trunc(swc_file_new_1.loc[index,'y'])
        new_1_trunc[index][2] = math.trunc(swc_file_new_1.loc[index,'z'])
        new_1_round[index][0] = round(swc_file_new_1.loc[index,'x'])
        new_1_round[index][1] = round(swc_file_new_1.loc[index,'y'])
        new_1_round[index][2] = round(swc_file_new_1.loc[index,'z'])

    for index in swc_file_new_2.index:
        new_2_trunc[index][0] = math.trunc(swc_file_new_2.loc[index,'x'])
        new_2_trunc[index][1] = math.trunc(swc_file_new_2.loc[index,'y'])
        new_2_trunc[index][2] = math.trunc(swc_file_new_2.loc[index,'z'])
        new_2_round[index][0] = round(swc_file_new_2.loc[index,'x'])
        new_2_round[index][1] = round(swc_file_new_2.loc[index,'y'])
        new_2_round[index][2] = round(swc_file_new_2.loc[index,'z']) 

    correct = 0
    incorrect = 0

    #iterate through old_1 and new_1 to find matches
    for i in range(0,len(swc_file_old_1)):    
        found_correct = 0
        for j in range(0,len(swc_file_new_1)):
            if old_1_trunc[i][0] == new_1_trunc[j][0] and old_1_trunc[i][1] == new_1_trunc[j][1] and old_1_trunc[i][2] == new_1_trunc[j][2]:
                correct = correct + 1
                found_correct = 1
            elif old_1_round[i][0] == new_1_round[j][0] and old_1_round[i][1] == new_1_round[j][1] and old_1_round[i][2] == new_1_round[j][2]:
                correct = correct + 1
                found_correct = 1   
        if found_correct == 0:
            incorrect = incorrect + 1

    #iterate through old_2 and new_2 to find matches
    for i in range(0,len(swc_file_old_2)):    
        found_correct = 0
        for j in range(0,len(swc_file_new_2)):
            if old_2_trunc[i][0] == new_2_trunc[j][0] and old_2_trunc[i][1] == new_2_trunc[j][1] and old_2_trunc[i][2] == new_2_trunc[j][2]:
                correct = correct + 1
                found_correct = 1
            elif old_2_round[i][0] == new_2_round[j][0] and old_2_round[i][1] == new_2_round[j][1] and old_2_round[i][2] == new_2_round[j][2]:
                correct = correct + 1
                found_correct = 1   
        if found_correct == 0:
            incorrect = incorrect + 1

    #print results
    length = len(swc_file_old_1) + len(swc_file_old_2)
    percent_correct = correct/length
    percent_incorrect = incorrect/length
    print("number correct: " + str(correct) + "/" + str(length))
    print("number incorrect: " + str(incorrect) + "/" + str(length))
    print("percent correct: " + str(percent_correct))
    print("percent incorrect: " + str(percent_incorrect))

    #save data
    total_correct = total_correct + correct
    total_incorrect = total_incorrect + incorrect
    total_length = total_length + length

    data_matrix[0][p-1] = correct
    data_matrix[1][p-1] = incorrect
    data_matrix[2][p-1] = percent_correct
    data_matrix[3][p-1] = percent_incorrect
    data_matrix[4][p-1] = length


average_percent_correct = total_correct/total_length
average_percent_incorrect = total_incorrect/total_length
print("average percent correct: " + str(average_percent_correct))
print("average percent incorrect: " + str(average_percent_incorrect))

#save data
np.save("C:\\Users\\chsha\\Downloads\\gcut\\gcut\\matlab\\demo_result\\data_matrix.npy", data_matrix,allow_pickle=True,fix_imports=True)







            

    





    
        
                






 
    
