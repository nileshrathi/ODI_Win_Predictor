import csv
import sys
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from scipy.special import softmax

#SOME CHECKED FACTS ABOUT DATA

#ALL ARE 50 OVR MATCHES
#THERE ARE CASES WHERE FIRST TEAM GOT ALL OUT BEFORE 50 OVR
#DATA has SOME ERRORS AS MAY 16 1999 MATCH WAS 50 OVR BUT IT SHOWS 49 OVR MATCH (OTHER ENTRIES LIKE THIS ARE ALSO THERE)
#IF MATCH REDUCED TO LESS THAN 50 OVERS, THIS INFO IS NOT THERE LIKE 31 AUGUST 2008 ENG VS SA MATCH (SO FAR I HAVE NOT CONSIDERED IT EITHER)
#STARTING EVERY NEW MATCH HAS A NEW GAME FIELD 1
#analysis for in terms of wickets-in-hand w and overs-to-go u


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1): #TO EXCLUDE THE LABEL
		# if(i==0 and row1[i]<=10 and row1[i]>7 ):
		# 	distance += ((row1[i] - row2[i])**2	)*4
		# if(i==0 and row1[i]<=7 and row1[i]>5 ):
		# 	distance += ((row1[i] - row2[i])**2	)*3
		# if(i==0 and row1[i]<=5 and row1[i]>3 ):
		# 	distance += ((row1[i] - row2[i])**2	)*2
		# if(i==0 and row1[i]<=3 and row1[i]>0 ):
		# 	distance += ((row1[i] - row2[i])**2	)*1
			
		distance += (row1[i] - row2[i])**2
	distance += (row1[0] - row2[0])**2
	return math.sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    #print(distances)
    neighbors = list()
    dist=list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    for i in range(num_neighbors):
        dist.append(distances[i][1])
    
    return neighbors,dist


# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors,dist = get_neighbors(train, test_row, num_neighbors)
    #print(neighbors)
    output_values = [row[-1] for row in neighbors]
    #prediction = max(set(output_values), key=output_values.count)
    return output_values,dist

def getAndProcessData():
	data = []
	data_inning1 = []
	data_inning2 = []
	with open('04_cricket_1999to2011.csv','rt') as fin:
		csv_data = csv.reader(fin)
		for row in csv_data:
			data.append(copy.deepcopy(row))
	ovr_wick_matrix = [[0 for col in range(11)] for row in range(51)]
	w_in_hand = 0
	o_to_go = 0
	curr_label = 0 #0 for 1st bat team win and 1 for 2nd bat team win
	curr_target = 0
	for i in range(1,len(data)):
		row = copy.deepcopy(data[i])
		#Second innings DATA
		if(row[2] == '2' and prev_inn == '1'):
			o_to_go = 50
			w_in_hand = 10
			#data_inning2.append([w_in_hand, float(curr_target) / float(o_to_go), curr_label])
			data_inning2.append([w_in_hand, float(curr_target), float(o_to_go), curr_label])
			curr_target = max(0, curr_target - int(row[4]))
			o_to_go = o_to_go - 1
			w_in_hand = int(row[11])
			if(curr_target > 0 and o_to_go > 0):
				#data_inning2.append([w_in_hand, float(curr_target) / float(o_to_go), curr_label])
				data_inning2.append([w_in_hand, float(curr_target), float(o_to_go), curr_label])
			prev_inn = row[2]
		elif(row[2] == '2'):
			o_to_go = o_to_go - 1
			w_in_hand = int(row[11])
			curr_target = max(0, curr_target - int(row[4]))
			if(curr_target > 0 and o_to_go > 0):
				#data_inning2.append([w_in_hand, float(curr_target) / float(o_to_go), curr_label])
				data_inning2.append([w_in_hand, float(curr_target), float(o_to_go), curr_label])
			prev_inn = row[2]
		#New game check
		if(row[35] == '1'):
			#DECIDE LABEL
			bat_first = str(row[18]).strip()
			win_team = str(row[25]).strip()
			if bat_first == win_team:
				curr_label = 0
			else:
				curr_label = 1
			#DECIDE TARGET
			curr_target = max(0, int(row[6])) #Precaution
		prev_inn = row[2]
	return data_inning1, data_inning2

data_inning1, data_inning2 = getAndProcessData()

#K-NEAREST NEIGHBOUR FOR SECOND INNING WIN PREDICTION

#TRAIN AND VALIDATION DIVISION (9:1)
random.shuffle(data_inning2)
train_data_inning2 = data_inning2[:int((len(data_inning2)+1)*.90)] # 90$ to train
valid_data_inning2 = data_inning2[int(len(data_inning2)*.90 + 1):] # 10% to validation
print('Train Size:', len(train_data_inning2))
print('Validation Size:', len(valid_data_inning2))

#number of neighbours
num_neighbors = 10


#MAKE PREDICTIONS
act_exp_vector = [] #actual and expected vector
for i in range(len(valid_data_inning2)):
	prediction_vector,dist = predict_classification(train_data_inning2, valid_data_inning2[i], num_neighbors)
	list(filter((0.0).__ne__, dist))
	num_neighbors=len(dist)
	maxdist=max(dist)
	newdist = dist-np.max(dist)
	m = softmax(newdist)*num_neighbors
	multiplicity=m
	prediction_vector2=np.multiply(prediction_vector,multiplicity)
	#multiplicity=[1.5,1.5,1.25,1.25,1.25,1.25,0.5,0.5,0.5,0.5]
	win_perentage = (sum(prediction_vector2) / float(len(prediction_vector2))) * 100.0
	#print('Actual:',valid_data_inning2[i][-1], 'Got:', win_perentage)
	act_exp_vector.append([valid_data_inning2[i][-1], win_perentage])

#CALCULATE ACCURACY
correct_count = 0
for i in range(len (act_exp_vector)):
	if(act_exp_vector[i][1] >= 50 and act_exp_vector[i][0] == 1):
		correct_count = correct_count + 1
	elif(act_exp_vector[i][1] < 50 and act_exp_vector[i][0] == 0):
		correct_count = correct_count + 1

print('Accuracy:',(correct_count / float(len(act_exp_vector))) * 100.0,'%')

'''

#TEST
test_vector = [8, 50.0, 20.0, 1.0]
prediction_vector = predict_classification(train_data_inning2, test_vector, num_neighbors)
win_perentage = (sum(prediction_vector) / float(len(prediction_vector))) * 100.0
print('Actual:',test_vector[-1], 'Got:', win_perentage)
'''

