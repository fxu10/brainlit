import scipy.io as sio
from scipy.linalg import norm
import math
from pathlib import Path
from mouselight_code.src import read_swc
import random
import numpy as np

for p in range(1,101):
    print("Number" + str(p))

    #select 2 random swc files
    swc_file_dir = Path("C:\\Users\\chsha\\Downloads\\JovoLAb\\consensus\\consensus")
    swc_files = list(swc_file_dir.glob("*.swc")) 

    #random swc 1
    swc_file_val = random.randrange(0,len(swc_files)) 
    swc_file_2_val = swc_file_val

    #make sure random swc 2 isn't the same as swc 1
    while swc_file_val == swc_file_2_val:
        swc_file_2_val = random.randrange(0,len(swc_files))


    swc_file = read_swc.read_swc(swc_files[swc_file_val])
    swc_file = swc_file[0]
    swc_file_2 = read_swc.read_swc(swc_files[swc_file_2_val])
    swc_file_2 = swc_file_2[0]
    print(len(swc_file))
    print(len(swc_file_2))

    #create mat file with data!!!
    mat_file = {}
    x = np.zeros(len(swc_file) + len(swc_file_2))
    y = np.zeros(len(swc_file) + len(swc_file_2))
    z = np.zeros(len(swc_file) + len(swc_file_2))
    D = np.zeros(len(swc_file) + len(swc_file_2)) #radius
    R = np.zeros(len(swc_file) + len(swc_file_2)) #structure
    parent = np.zeros(len(swc_file) + len(swc_file_2))
    sample = np.zeros(len(swc_file) + len(swc_file_2))

    for index in swc_file.index:  
        x[index] = swc_file.loc[index, 'x']
        y[index] = swc_file.loc[index,'y']
        z[index] = swc_file.loc[index,'z']
        D[index] = swc_file.loc[index,'r']
        R[index] = swc_file.loc[index, 'structure']
        sample[index] = swc_file.loc[index,'sample'] - 1 
        if swc_file.loc[index,'parent'] == -1:
            parent[index] = -1
        if swc_file.loc[index,'parent'] != -1:
            parent[index] = swc_file.loc[index,'parent'] - 1

    for index in swc_file_2.index:
        x[index+len(swc_file)] = swc_file_2.loc[index, 'x']
        y[index+len(swc_file)] = swc_file_2.loc[index,'y']
        z[index+len(swc_file)] = swc_file_2.loc[index,'z']
        D[index+len(swc_file)] = swc_file_2.loc[index,'r']
        R[index+len(swc_file)] = swc_file_2.loc[index, 'structure']
        sample[index+len(swc_file)] = swc_file_2.loc[index, 'sample'] + len(swc_file) - 1
        #parent index - but exclude no parent (somas)
        if swc_file_2.loc[index,'parent'] == -1:
            parent[index+len(swc_file)] = -1
        if swc_file_2.loc[index,'parent'] != -1:
            parent[index+len(swc_file)] = swc_file_2.loc[index, 'parent'] + len(swc_file) - 1
 
    mat_file['x'] = np.matrix(x).T
    mat_file['y'] = np.matrix(y).T
    mat_file['z'] = np.matrix(z).T
    mat_file['D'] = np.matrix(D).T
    mat_file['R'] = np.matrix(R).T

    #create dA with existing connections
    dA = np.zeros((len(swc_file) + len(swc_file_2),len(swc_file) + len(swc_file_2)))

    for i in range(0,len(dA)):
        if parent[i] != -1:
            dA[int(sample[i])][int(parent[i])] = 1

    #save soma_ind.txt
    save_soma_ind_path = "C:\\Users\\chsha\\Downloads\\gcut\\gcut\\matlab\\demo_data\\mat_file_" + str(p) + "_soma_ind.txt"
    soma_ind = open(save_soma_ind_path, "w")
    for i in range(len(parent)):
        if parent[i] == -1:
            soma = str(sample[i] + 1)
            soma_ind.write(soma  + "\n")
    soma_ind.close()

    #put important info into .txt file
    imp_info_path = "C:\\Users\\chsha\\Downloads\\gcut\\gcut\\matlab\\demo_data\\imp_info\\imp_info_" + str(p) + ".txt"
    imp_info = open(imp_info_path, "w")
    imp_info.write(str(swc_files[swc_file_val])+ "\n")
    imp_info.write(str(swc_files[swc_file_2_val]) + "\n")
    for i in range(len(parent)):
        if parent[i] == -1:
            soma = str(sample[i] + 1)
            imp_info.write(soma  + "\n")
    

    #now, add an edge
    proximity_bias = 1
    number_of_new_edges = 1
    imp_info.write(str(number_of_new_edges)  + "\n")

    if proximity_bias == 0:
        for i in range(0,number_of_new_edges):
            added_edge = 0
            while added_edge == 0:
                #random point on swc_file
                node1 = random.randrange(0,len(swc_file))
                node2 = random.randrange(len(swc_file),len(swc_file) + len(swc_file_2))
                #making sure you don't repeat edges
                if dA[node1][node2] != 1 and dA[node2][node1] != 1:
                    dA[node1][node2] = 1
                    added_edge = 1
                    print("added an edge between " + str(node1) + " and " + str(node2))
                    imp_info.write(str(node1 + 1) + "\n")
                    imp_info.write(str(node2 + 1) + "\n")

    if proximity_bias == 1:
        #first create probability distribution
        print("proximity bias 1")
        sum_probs = 0
        prob_dist = np.zeros((len(swc_file),len(swc_file_2)))

        for i in range(0,len(swc_file)):
            for j in range(0,len(swc_file_2)):
                added_value = math.exp(-1*np.linalg.norm(np.array((x[i],y[i],z[i]))-np.array((x[j+len(swc_file)],y[j+len(swc_file)],z[j+len(swc_file)]))))
                prob_dist[i][j] = added_value
                sum_probs = sum_probs + added_value
        
        print("created prob dist")

        prob_dist = np.true_divide(prob_dist,sum_probs)


        for i in range(0,number_of_new_edges):
            added_edge = 0
            while added_edge == 0:
                #select random node
                node1 = random.randrange(0,len(swc_file)+len(swc_file_2))

                #case of node1 being in 1st neuron
                if node1 < len(swc_file):
                    max_prob = 0
                    node2_index = 0

                    for i in range(0,len(swc_file_2)):
                        if prob_dist[node1][i] > max_prob:
                            max_prob = prob_dist[node1][i]
                            node2_index = i
                        
                    if max_prob > 0 and dA[node1][node2_index+len(swc_file)] == 0 and dA[node2_index+len(swc_file)][node1] == 0:
                        dA[node1][node2_index+len(swc_file)] = 1
                        print("case 1 neuron")
                        print("added edge between " + str(node1) + " and " +str(node2_index+len(swc_file)))
                        imp_info.write(str(node1 + 1) + "\n")
                        imp_info.write(str(node2_index + len(swc_file) + 1) + "\n")
                        added_edge = 1


                #case of node1 being in second neuron
                if node1 >= len(swc_file):
                    node1 = node1 - len(swc_file)
                    max_prob = 0
                    node2_index = 0

                    for i in range(0,len(swc_file)):
                        if prob_dist[i][node1] > max_prob:
                            max_prob = prob_dist[i][node1]
                            node2_index = i
                    
                    if max_prob > 0 and dA[node2_index,node1+len(swc_file)] == 0 and dA[node1+len(swc_file),node2_index] == 0:
                        dA[node2_index][node1+len(swc_file)] = 1
                        print("case 2nd neuron")
                        print("added edge between " + str(node2_index) + " and " + str(node1+len(swc_file)))
                        imp_info.write(str(node2_index + 1) + "\n")
                        imp_info.write(str(node1 + len(swc_file)+ 1) + "\n")
                        added_edge = 1
    
    mat_file['dA'] = dA
    imp_info.close()

    save_mat_path = 'C:\\Users\\chsha\\Downloads\\gcut\\gcut\\matlab\\demo_data\\mat_file_' + str(p) + '.mat'   
    sio.savemat(save_mat_path, mat_file)


    















            

            










        




    

