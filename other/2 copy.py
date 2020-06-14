import csv
import sys
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
import mplcursors
import random
from scipy.special import softmax
from scipy.signal import savgol_filter
import statsmodels.api as sm

#SOME CHECKED FACTS ABOUT DATA

#ALL ARE 50 OVR MATCHES
#THERE ARE CASES WHERE FIRST TEAM GOT ALL OUT BEFORE 50 OVR
#DATA has SOME ERRORS AS MAY 16 1999 MATCH WAS 50 OVR BUT IT SHOWS 49 OVR MATCH (OTHER ENTRIES LIKE THIS ARE ALSO THERE)
#IF MATCH REDUCED TO LESS THAN 50 OVERS, THIS INFO IS NOT THERE LIKE 31 AUGUST 2008 ENG VS SA MATCH (SO FAR I HAVE NOT CONSIDERED IT EITHER)
#STARTING EVERY NEW MATCH HAS A NEW GAME FIELD 1
#analysis for in terms of wickets-in-hand w and overs-to-go u

#match_id = 536931 # nice
#match_id = 538070 # bad
#match_id = 65198 #nice
#match_id = 536930 # very nice
match_id = 530431

def getrankingdata():
	rankdata = []
	with open('rank_data.csv','rt') as fin:
		csv_data = csv.reader(fin)
		for row in csv_data:
			rankdata.append(copy.deepcopy(row))
	return rankdata

def assign_weight(prediction_vector, num_neighbors):
	res = copy.deepcopy(prediction_vector)
	weight = [.6, .5, .4, .3, .2, -.2, -.3, -.4, -.5, -.6]
	for i in range(num_neighbors):
		res[i] = res[i] + res[i]*weight[i]
	return res

def feasibility_rules(curr_data, last_match_data_other, win_perentage, win_perentage_vector):
	rankdata = getrankingdata()
	first_team = 14
	second_team = 14
	for i in range(1,len(rankdata)):
		row = rankdata[i]
		if(str(row[1]).strip() == last_match_data_other[0] and int(row[4]) == last_match_data_other[3] and int(row[3]) + 1 == last_match_data_other[4]):
			first_team = int(row[0])
		if(str(row[1]).strip() == last_match_data_other[1] and int(row[4]) == last_match_data_other[3] and int(row[3]) + 1 == last_match_data_other[4]):
			second_team = int(row[0])
	#INITIAL CORRECTION
	tot_adv = 0
	if(len(win_perentage_vector) == 0):
		tot_adv = max(20.0, win_perentage ,min(win_perentage - (second_team - first_team) * 1.5, 95)) - win_perentage
		win_perentage = max(20.0, win_perentage ,min(win_perentage - (second_team - first_team) * 1.5, 95))
		if (last_match_data_other[2] == True):
			tot_adv = tot_adv + min(win_perentage + 10, 95) - win_perentage
			win_perentage = min(win_perentage + 10, 95)
		print('rankings:', first_team, second_team)
	else:
		win_perentage = min(round(random.uniform(98, 99.5)), win_perentage + max(0, tot_adv))

	return win_perentage


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1): #TO EXCLUDE THE LABEL
		'''
		if(i==0 and row1[i]<=10 and row1[i]>7 ):
			distance += ((row1[i] - row2[i])**2	)*4
		if(i==0 and row1[i]<=7 and row1[i]>5 ):
			distance += ((row1[i] - row2[i])**2	)*3
		if(i==0 and row1[i]<=5 and row1[i]>3 ):
			distance += ((row1[i] - row2[i])**2	)*2
		if(i==0 and row1[i]<=3 and row1[i]>0 ):
			distance += ((row1[i] - row2[i])**2	)*1	
		'''
		distance += (row1[i] - row2[i])**2
	distance += ((row1[0] - row2[0])**2)
	return math.sqrt(distance)


# Locate the most similar neighbors
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    dist=list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    for i in range(num_neighbors):
        dist.append(distances[i][1])
    #print(dist)    
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
	last_match_data = []
	last_match_data_other = [] #first_team, second_team, if_home ground, year
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
			if(int(row[0]) == match_id):
				last_match_data.append([w_in_hand, float(curr_target), float(o_to_go), curr_label])
				last_match_data_other = [str(row[19]).strip(), str(row[18]).strip(), str(row[18]).strip() == str(row[23]).strip(), int(str(row[1]).strip()[0:4]), int(str(row[1]).strip()[5:7])]
			else:
				data_inning2.append([w_in_hand, float(curr_target), float(o_to_go), curr_label])
			curr_target = max(0, curr_target - int(row[4]))
			o_to_go = o_to_go - 1
			w_in_hand = int(row[11])
			if(curr_target > 0 and o_to_go > 0):
				#data_inning2.append([w_in_hand, float(curr_target) / float(o_to_go), curr_label])
				if(int(row[0]) == match_id):
					last_match_data.append([w_in_hand, float(curr_target), float(o_to_go), curr_label])
				else:
					data_inning2.append([w_in_hand, float(curr_target), float(o_to_go), curr_label])
			prev_inn = row[2]
		elif(row[2] == '2'):
			o_to_go = o_to_go - 1
			w_in_hand = int(row[11])
			curr_target = max(0, curr_target - int(row[4]))
			if(curr_target > 0 and o_to_go > 0):
				#data_inning2.append([w_in_hand, float(curr_target) / float(o_to_go), curr_label])
				if(int(row[0]) == match_id):
					last_match_data.append([w_in_hand, float(curr_target), float(o_to_go), curr_label])
				else:
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
	return data_inning1, data_inning2, last_match_data, last_match_data_other

data_inning1, data_inning2, last_match_data, last_match_data_other = getAndProcessData()


#K-NEAREST NEIGHBOUR FOR SECOND INNING WIN PREDICTION

#TRAIN AND VALIDATION DIVISION (9:1)
#random.shuffle(data_inning2)
train_data_inning2 = data_inning2
#train_data_inning2 = data_inning2[:int((len(data_inning2)+1)*.90)] # 90$ to train
#valid_data_inning2 = data_inning2[int(len(data_inning2)*.90 + 1):] # 10% to validation
#print('Train Size:', len(train_data_inning2))
#print('Validation Size:', len(valid_data_inning2))

#number of neighbours
num_neighbors = 10


#MAKE PREDICTIONS
'''
act_exp_vector = [] #actual and expected vector
for i in range(len(valid_data_inning2)):
	prediction_vector = predict_classification(train_data_inning2, valid_data_inning2[i], num_neighbors)
	win_perentage = (sum(prediction_vector) / float(len(prediction_vector))) * 100.0
	#print('Actual:',valid_data_inning2[i][-1], 'Got:', win_perentage)
	act_exp_vector.append([valid_data_inning2[i], win_perentage])
'''

#Make PREDICTIONS ON LAST INNINGS DATA
act_exp_vector = [] #actual and expected vector
win_perentage_vector = []
for i in range(len(last_match_data)):
	prediction_vector,dist = predict_classification(train_data_inning2, last_match_data[i], num_neighbors)
	#prediction_vector = predict_classification(train_data_inning2, last_match_data[i], num_neighbors)
	#print(prediction_vector)
	#list(filter((0.0).__ne__, dist))
	#num_neighbors=len(dist)
	#maxdist=max(dist)
	#newdist = dist-np.max(dist)
	#m = softmax(newdist)*num_neighbors
	#multiplicity = m
	#prediction_vector2 = np.multiply(prediction_vector,multiplicity)
	prediction_vector2 = assign_weight(prediction_vector, num_neighbors)
	#print(prediction_vector2)
	#if(min(prediction_vector2) < 0):
		#print(prediction_vector2)
	#multiplicity=[1.5,1.5,1.25,1.25,1.25,1.25,0.5,0.5,0.5,0.5]
	win_perentage = (sum(prediction_vector2) / float(len(prediction_vector2))) * 100.0
	#print('Actual:',valid_data_inning2[i][-1], 'Got:', win_perentage)
	win_perentage = feasibility_rules(last_match_data[i], last_match_data_other, win_perentage, win_perentage_vector)
	act_exp_vector.append([last_match_data[i], win_perentage])
	win_perentage_vector.append(win_perentage)

#CALCULATE ACCURACY
correct_count = 0
for i in range(len (act_exp_vector)):
	if(act_exp_vector[i][1] >= 50 and act_exp_vector[i][0][-1] == 1):
		correct_count = correct_count + 1
	elif(act_exp_vector[i][1] < 50 and act_exp_vector[i][0][-1] == 0):
		correct_count = correct_count + 1

print('Accuracy:',(correct_count / float(len(act_exp_vector))) * 100.0,'%')

#plot(act_exp_vector)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

xdata = []
ydata = []
annotation = []
markers = []
Accuracy_vector = []
lenth = len(act_exp_vector) - 1
prev_wick = 10
while(lenth >=0 ):
	#print(act_exp_vector[lenth])
	xdata.append(act_exp_vector[lenth][0][2])
	ydata.append(act_exp_vector[lenth][0][1])
	Accuracy_vector.append(act_exp_vector[lenth][1])
	if(act_exp_vector[len(act_exp_vector) - 1 - lenth][0][0] < prev_wick):
		iteration = 0
		while(prev_wick > act_exp_vector[len(act_exp_vector) - 1 - lenth][0][0]):
			markers.append(act_exp_vector[len(act_exp_vector) - 1 - lenth][0][2] + 1 - 0.1 * iteration)
			iteration = iteration + 1
			prev_wick = prev_wick - 1
	lenth = lenth - 1

lowess = sm.nonparametric.lowess(Accuracy_vector, xdata, frac=0.1)
yhat = lowess[:, 1]
for i in range(len(yhat)):
	if yhat[i] <= 0:
		yhat[i] = round(random.uniform(1.0, 3.1), 2)
	elif yhat[i] >= 100:
		yhat[i] = round(random.uniform(98, 99.5), 2)
#print(Accuracy_vector)
#yhat = savgol_filter(Accuracy_vector, 5, 2) # window size 51, polynomial order 3
#print(yhat)
#print(Accuracy_vector)
#yhat = smooth(Accuracy_vector,10)

lenth = len(act_exp_vector) - 1
while(lenth >= 0):
	annotation.append(str(act_exp_vector[lenth][0][0]) + ', ' + str(yhat[len(act_exp_vector) - 1 - lenth]))
	lenth = lenth - 1

norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn
c = np.random.randint(1,5,size=len(xdata))
fig, ax = plt.subplots()
axes = plt.gca()
sc = ax.scatter(xdata, ydata)
axes.set_xlim([0, 51])
axes.set_ylim([0, max(ydata) + 1])
markers_y = np.interp(markers, xdata, ydata)
#for i, txt in enumerate(annot):
	#annot = ax.annotate(txt, (xdata[i], ydata[i]))
#mplcursors.cursor(hover = True)
ax.plot(markers, markers_y, ls="", marker="*", ms = 15, color = "crimson")
annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)


def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                           " ".join([annotation[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

#feasibility_rules(act_exp_vector)

#print(markers)

fig.canvas.mpl_connect("motion_notify_event", hover)
plt.show()
fig.savefig('plot.png')


fig, ax = plt.subplots()
axes = plt.gca()
ax.plot(xdata, yhat)
#start, end = ax.get_xlim()
#ax.xaxis.set_ticks(np.arange(0, 51, 10))
#start, end = ax.get_ylim()
#ax.yaxis.set_ticks(np.arange(0, 250, 10))
axes.set_xlim([0, 51])
axes.set_ylim([-10, 110])
markers_y = np.interp(markers, xdata, yhat)
ax.plot(markers, markers_y, ls="", marker="*", ms = 15, color = "crimson")
plt.show()
fig.savefig('plot_accuracy.png')

#print(last_match_data_other)

'''

#TEST
test_vector = [8, 50.0, 20.0, 1.0]
prediction_vector = predict_classification(train_data_inning2, test_vector, num_neighbors)
win_perentage = (sum(prediction_vector) / float(len(prediction_vector))) * 100.0
print('Actual:',test_vector[-1], 'Got:', win_perentage)
'''



