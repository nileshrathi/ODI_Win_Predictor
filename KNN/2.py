import csv
import sys
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
import mplcursors
import random
import tkinter as tk
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

match_id = 536931 # nice
#match_id = 538070 # bad
#match_id = 65198 #nice
#match_id = 536930 # very nice
#match_id = 530431

#gui = False
gui = True

def getrankingdata():
	rankdata = []
	with open('rank_data.csv','rt') as fin:
		csv_data = csv.reader(fin)
		for row in csv_data:
			rankdata.append(copy.deepcopy(row))
	return rankdata

def assign_weight(prediction_vector):
	res = copy.deepcopy(prediction_vector)
	weight = [0 for i in range(len(prediction_vector))]
	i = 0
	j = len(prediction_vector) - 1
	perc = .6
	while i < j:
		weight[i] = perc
		weight[j] = 0 - perc
		i = i + 1
		j = j - 1
		perc = max(.1, perc - .1)
	for i in range(len(prediction_vector)):
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
	if(not (curr_data[0] < 3 and curr_data[2] < 10 and curr_data[1]/curr_data[2] > 10)):
		if(len(win_perentage_vector) == 0 and curr_data[2] >= 10):
			tot_adv = max(win_perentage ,min(win_perentage - (second_team - first_team) * 1.5, 99)) - win_perentage
			win_perentage = max(win_perentage, min(win_perentage - (second_team - first_team) * 1.5, 99))
			if (last_match_data_other[2] == True):
				tot_adv = tot_adv + min(win_perentage + (win_perentage * .2), 99) - win_perentage
				win_perentage = min(win_perentage + (win_perentage * .2), 99)
			print('rankings:', first_team, second_team)
		else:
			win_perentage = min(round(random.uniform(98, 99.5)), win_perentage + max(0, tot_adv))
	if(win_perentage <= 0):
		win_perentage = round(random.uniform(0.1, 1.1), 2)
	return round(win_perentage, 2)


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

def smooth(y, box_pts):
	box = np.ones(box_pts)/box_pts
	y_smooth = np.convolve(y, box, mode='same')
	return y_smooth

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

if gui == False:
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
		prediction_vector, dist = predict_classification(train_data_inning2, last_match_data[i], num_neighbors)
		#prediction_vector = predict_classification(train_data_inning2, last_match_data[i], num_neighbors)
		#print(prediction_vector)
		#list(filter((0.0).__ne__, dist))
		#num_neighbors=len(dist)
		#maxdist=max(dist)
		#newdist = dist-np.max(dist)
		#m = softmax(newdist)*num_neighbors
		#multiplicity = m
		#prediction_vector2 = np.multiply(prediction_vector,multiplicity)
		prediction_vector2 = assign_weight(prediction_vector)
		#print(prediction_vector2)
		#if(min(prediction_vector2) < 0):
			#print(prediction_vector2)
		#multiplicity=[1.5,1.5,1.25,1.25,1.25,1.25,0.5,0.5,0.5,0.5]
		win_perentage = (sum(prediction_vector2) / float(len(prediction_vector2))) * 100.0
		#print('Actual:',valid_data_inning2[i][-1], 'Got:', win_perentage)
		win_perentage = feasibility_rules(last_match_data[i], last_match_data_other, win_perentage, win_perentage_vector)
		act_exp_vector.append([last_match_data[i], win_perentage])
		win_perentage_vector.append(win_perentage)

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
	axes.set_xlabel('Overs To Go (OG)')
	axes.set_ylabel('Remaining Target (RT)')
	axes.set_xlim([0, 51])
	axes.set_ylim([0, max(ydata) + 1])
	markers_y = np.interp(markers, xdata, ydata)
	#for i, txt in enumerate(annot):
		#annot = ax.annotate(txt, (xdata[i], ydata[i]))
	#mplcursors.cursor(hover = True)
	ax.plot(markers, markers_y, ls="", marker="*", ms = 15, color = "crimson", label = 'Fall of Wicket')
	plt.legend(loc = 'upper left')
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

if gui == False:
	fig.canvas.mpl_connect("motion_notify_event", hover)
	plt.show()
	fig.savefig('plot.png')
	fig, ax = plt.subplots()
	axes = plt.gca()
	
	nn_536931 = [58.627880909403814, 60.228543039095605, 54.94359808453059, 55.1444764263862, 51.44859483506944, 50.49944097345526, 51.69628848396771, 50.63931419735863, 52.42817038356668, 38.01060007579291, 35.696778908754005, 35.03071618458581, 37.88780212402344, 38.51843747225675, 40.68849100869083, 42.44467070608428, 34.478769302368164, 35.841891381048384, 32.52651065975041, 28.817138671875, 32.55086695894282, 36.04491967949078, 35.575211796447306, 33.26976634837963, 32.62385441706731, 30.69985866546631, 30.993116425304876, 26.327273259397412, 24.854763715695114, 26.37125977682411, 30.695443755214654, 31.992923245571628, 37.942032509661736, 44.573565954449535, 43.71064129997702, 44.60748931508005, 51.54170833221853, 51.872798374720986, 69.21676368044135, 78.16612243652344, 85.4816722869873, 97.03666340221058, 62.4834317939226, 99.98, 74.48755536760602, 99.98, 99.98, 99.98]
	nn_536931.reverse()
	
	#nn_538070 = [99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 93.4805746639476, 93.60288902147282, 59.29736291577002, 61.83915737860217, 59.2530898105951, 54.48687061759996, 49.88321304321289, 47.84854986728766, 46.17183562247984, 46.89494612949346, 48.44945440229201, 50.389324418650375, 50.44628373728502, 52.73134414463827, 54.89075351768816, 54.8321533203125, 59.23244591915246, 58.381335918719955, 64.13562869237474, 66.44257113464877, 68.66419969406803, 67.6774297441755, 78.06842280369179, 76.1330481803063, 79.37088012695312, 78.63818296452159, 80.02012536880818, 82.236328125, 81.66796077381481, 86.69082457760732, 85.97527963143807, 89.31725150660463, 90.26222766285211, 87.66128540039062, 76.34022620416457, 62.9890803967492, 57.94957941228694, 50.40119404695472, 33.993601697556514, 29.11710790685705, 24.078631767859825, 2.0, 5.0, 2.0]
	#nn_538070.reverse()
	#del nn_538070[-1]

	#nn_65198 = [90.09401239900507, 89.6923486117659, 79.70297540937152, 78.43183653695243, 83.45001796506486, 83.54585920061383, 90.61067199707031, 92.5934829711914, 96.25302186946278, 99.98, 99.98, 96.65408750123615, 98.42764166337025, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98]
	#nn_65198.reverse()
	#del nn_65198[-1]

	#nn_536930 = [54.561733669704864, 54.04035818264279, 52.54041169819079, 53.99287803500306, 53.75316127654045, 52.159646223431174, 50.961534149792726, 43.79677293706237, 41.609446207682296, 41.51139609834068, 42.75237358940973, 41.62575897079947, 43.105043481897425, 44.66024590451181, 45.761407501068874, 45.44752854567308, 46.06333686706216, 28.95274394267314, 30.351703086595858, 28.5178617997603, 26.84057158541817, 25.41912914027829, 25.040176768361786, 21.624817082911363, 23.9584330639808, 24.84459232639622, 29.309916324752695, 30.977568449797456, 33.85464594914363, 36.20427449544271, 42.08372083760924, 45.07871928967928, 47.51964915882457, 47.6036141771789, 50.86087908063616, 58.43711385921556, 64.514405992296, 68.85636492473323, 68.6117172241211, 68.74375062830308, 68.43957265218098, 58.582600376062224, 49.06791111208358, 38.12180360158284, 50.01099326393821, 79.56815802532694, 62.668094635009766, 44.31262969970703, 69.75403785705566, 2.0]
	#nn_536930.reverse()
	#del nn_536930[-1]
	
	#nn_530431 = [82.72334965396689, 82.13423618050508, 84.8209511361471, 82.0929178377477, 78.94278177400915, 75.96195872237043, 74.96915750835666, 74.67853252704327, 72.21947738804768, 73.17688778892082, 72.75887022627161, 71.56018231366131, 73.45917615023527, 74.83998857306305, 69.97156931254678, 70.85653987931616, 70.47128025489518, 70.05671315378957, 69.08031135205401, 70.65800136990018, 70.15213012695312, 70.6768798828125, 74.07222202845982, 71.91564006190146, 72.20712694071106, 78.45068613688152, 63.2816613889208, 60.07785423129212, 59.39944488833649, 56.691907115818296, 60.82642801691977, 59.17806403581486, 61.28281593322754, 60.20437387319711, 63.57881728916952, 66.7226929595505, 76.04768814579133, 93.22590378095518, 99.98, 98.52353731791179, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98]
	#nn_530431.reverse()
	#del nn_530431[-1]

	nn_536931_dl = [58.627880909403814, 60.228543039095605, 54.94359808453059, 55.1444764263862, 51.44859483506944, 50.49944097345526, 51.69628848396771, 50.63931419735863, 52.42817038356668, 38.01060007579291, 35.696778908754005, 35.03071618458581, 37.88780212402344, 38.51843747225675, 40.68849100869083, 42.44467070608428, 34.478769302368164, 35.841891381048384, 32.52651065975041, 28.817138671875, 32.55086695894282, 36.04491967949078, 35.575211796447306, 33.26976634837963, 32.62385441706731, 30.69985866546631, 30.993116425304876, 26.327273259397412, 24.854763715695114, 26.37125977682411, 30.695443755214654, 31.992923245571628, 37.942032509661736, 44.573565954449535, 43.71064129997702, 44.60748931508005, 51.54170833221853, 51.872798374720986, 69.21676368044135, 78.16612243652344, 43.20315567015299, 48.475911584033426, 9.918101348072064, 34.156394184899455, 1.0, 17.969541158687427, 76.52219327774863, 5.0]
	nn_536931_dl.reverse()
	#del nn_536931_dl[-1]
	
	#nn_538070_dl = [99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 93.4805746639476, 93.60288902147282, 59.29736291577002, 61.83915737860217, 59.2530898105951, 54.48687061759996, 49.88321304321289, 47.84854986728766, 46.17183562247984, 46.89494612949346, 48.44945440229201, 50.389324418650375, 50.44628373728502, 52.73134414463827, 54.89075351768816, 54.8321533203125, 59.23244591915246, 58.381335918719955, 64.13562869237474, 66.44257113464877, 68.66419969406803, 67.6774297441755, 78.06842280369179, 76.1330481803063, 79.37088012695312, 78.63818296452159, 80.02012536880818, 82.236328125, 81.66796077381481, 86.69082457760732, 85.97527963143807, 89.31725150660463, 90.26222766285211, 87.66128540039062, 63.20699607890681, 50.079245295560646, 48.33321965471441, 49.121593569677145, 40.30785973181092, 46.3266455215997, 58.01081997174597, 48.434761561053534, 99.0, 2.0]
	#nn_538070_dl.reverse()
	#del nn_538070_dl[-1]

	#nn_65198_dl = [90.09401239900507, 89.6923486117659, 79.70297540937152, 78.43183653695243, 83.45001796506486, 83.54585920061383, 90.61067199707031, 92.5934829711914, 96.25302186946278, 99.98, 99.98, 96.65408750123615, 98.42764166337025, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98, 99.98]
	#nn_65198_dl.reverse()
	#del nn_65198_dl[-1]

	#nn_536930_dl = [54.561733669704864, 54.04035818264279, 52.54041169819079, 53.99287803500306, 53.75316127654045, 52.159646223431174, 50.961534149792726, 43.79677293706237, 41.609446207682296, 41.51139609834068, 42.75237358940973, 41.62575897079947, 43.105043481897425, 44.66024590451181, 45.761407501068874, 45.44752854567308, 46.06333686706216, 28.95274394267314, 30.351703086595858, 28.5178617997603, 26.84057158541817, 25.41912914027829, 25.040176768361786, 21.624817082911363, 23.9584330639808, 24.84459232639622, 29.309916324752695, 30.977568449797456, 33.85464594914363, 36.20427449544271, 42.08372083760924, 45.07871928967928, 47.51964915882457, 47.6036141771789, 50.86087908063616, 58.43711385921556, 64.514405992296, 68.85636492473323, 68.6117172241211, 68.74375062830308, 68.43957265218098, 80.60570093996652, 78.58347957813866, 77.983545813113, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0]
	#nn_536930_dl.reverse()
	#del nn_536930_dl[-1]
	
	#nn_530431_dl = [82.72334965396689, 82.13423618050508, 84.8209511361471, 82.0929178377477, 78.94278177400915, 75.96195872237043, 74.96915750835666, 74.67853252704327, 72.21947738804768, 73.17688778892082, 72.75887022627161, 71.56018231366131, 73.45917615023527, 74.83998857306305, 69.97156931254678, 70.85653987931616, 70.47128025489518, 70.05671315378957, 69.08031135205401, 70.65800136990018, 70.15213012695312, 70.6768798828125, 74.07222202845982, 71.91564006190146, 72.20712694071106, 78.45068613688152, 63.2816613889208, 60.07785423129212, 59.39944488833649, 56.691907115818296, 60.82642801691977, 59.17806403581486, 61.28281593322754, 60.20437387319711, 63.57881728916952, 66.7226929595505, 76.04768814579133, 93.22590378095518, 99.98, 98.52353731791179, 99.98, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0]
	#nn_530431_dl.reverse()
	#del nn_530431_dl[-1]

	nn = nn_536931
	#nn = nn_538070
	#nn = nn_65198
	#nn = nn_536930
	#nn = nn_530431

	nn_dl = nn_536931_dl
	#nn_dl = nn_538070_dl
	#nn_dl = nn_65198_dl
	#nn_dl = nn_536930_dl
	#nn_dl = nn_530431_dl

	'''
	knn_nn_avg = []
	for i in range(len(nn)):
		knn_nn_avg.append((nn[i] + nn_dl[i] + yhat[i])/3.0)
	'''
	#ax.plot(xdata, yhat, label = 'Modified KNN')
	ax.plot(xdata, nn, label = 'Neural network(NN)')
	ax.plot(xdata, nn_dl, label = 'NN + D/L')
	
	#start, end = ax.get_xlim()
	#ax.xaxis.set_ticks(np.arange(0, 51, 10))
	#start, end = ax.get_ylim()
	#ax.yaxis.set_ticks(np.arange(0, 250, 10))
	axes.set_xlim([0, 51])
	axes.set_ylim([0, 110])
	axes.set_xlabel('Overs To Go (OG)')
	axes.set_ylabel('Win Probabilitiy (%)')
	markers_y = np.interp(markers, xdata, yhat)
	markers_y1 = np.interp(markers, xdata, nn)
	markers_y2 = np.interp(markers, xdata, nn_dl)
	
	#ax.plot(markers, markers_y, ls="", marker="*", ms = 15, color = "crimson")
	ax.plot(markers, markers_y1, ls="", marker="*", ms = 15, color = "crimson")
	ax.plot(markers, markers_y2, ls="", marker="*", ms = 15, color = "crimson")

	#for x in markers:
		#ax.axvline(x, color = 'limegreen')
	plt.legend(loc = 'upper right')
	plt.show()
	fig.savefig('plot_accuracy.png')


def WINPredictor(entries):
	Wickets_in_Hand= (int(entries['Wickets_in_Hand'].get()) )
	Overs_Remaining= (int(entries['Overs_Remaining'].get()) )
	Target_Score= (int(entries['Current_Target_Score'].get()) )
	Number_of_neighbours_for_KNN= (int(entries['Number_of_neighbours_for_KNN'].get()) )
	Year_of_ODI= (int(entries['Year_of_ODI'].get()) )
	First_Bat_Team= (str(entries['First_Bat_Team'].get()) )
	Second_Bat_Team= (str(entries['Second_Bat_Team'].get()) )
	Home_ground_Second_Bat_Team= (int(entries['Home_ground_Second_Bat_Team_(1/0)'].get()))
	Month_of_ODI = (int(entries['Month_of_ODI'].get()) )
	#print(Wickets_in_Hand,Overs_Remaining,Target_Score,Number_of_neighbours_for_KNN,Year_of_ODI,First_Bat_Team,Second_Bat_Team,Home_ground_Second_Bat_Team)
	#PREDICTION
	prediction_vector,dist = predict_classification(train_data_inning2, [Wickets_in_Hand, float(Target_Score), float(Overs_Remaining), 0], Number_of_neighbours_for_KNN)
	#print(dist)
	prediction_vector1 = []
	for j in range(len(dist)):
		if dist[j] <= 3.0 and (float(Target_Score)/float(Overs_Remaining) <= 25):
			prediction_vector1.append(prediction_vector[j])
	if(len(prediction_vector1) != 0):
		prediction_vector2 = assign_weight(prediction_vector1)
		win_perentage = (sum(prediction_vector2) / float(len(prediction_vector2))) * 100.0
		win_perentage = feasibility_rules([Wickets_in_Hand, float(Target_Score), float(Overs_Remaining), 0], [First_Bat_Team, Second_Bat_Team, Home_ground_Second_Bat_Team, Year_of_ODI, Month_of_ODI], win_perentage, [])
		root = tk.Tk()
		root.title('Win Prediction')
		T = tk.Text(root,width = 40, height=4, font=("Helvetica", 32))
		T.pack()
		T.insert(tk.END, str(win_perentage) + '%')
		root.mainloop()
	else:
		root = tk.Tk()
		root.title('Win Prediction')
		T = tk.Text(root,width = 40, height=4, font=("Helvetica", 32))
		T.pack()
		T.insert(tk.END, 'HARD LUCK BUDDY! No closer situations found.')
		root.mainloop()


def makeform(root, fields):
	entries = {}
	for field in fields:
		print(field)
		row = tk.Frame(root)
		lab = tk.Label(row, width=50,text=field+": ", anchor='w')
		lab.config(font=("Courier", 20))
		ent = tk.Entry(row,font=("Helvetica", 32))
		if(field=='Wickets_in_Hand'):
			ent.insert(0, "8")
		if(field=='Overs_Remaining'):
			ent.insert(0, "20")
		if(field=='Current_Target_Score'):
			ent.insert(0, "110")
		if(field=='Number_of_neighbours_for_KNN'):
			ent.insert(0, "10")
		if(field=='Year_of_ODI'):
			ent.insert(0, "2008")
		if(field=='First_Bat_Team'):
			ent.insert(0, "West Indies")
		if(field=='Second_Bat_Team'):
			ent.insert(0, "India")
		if(field=='Home_ground_Second_Bat_Team_(1/0)'):
			ent.insert(0, "1")
		if(field=='Month_of_ODI'):
			ent.insert(0, "07")
		row.pack(side=tk.TOP, 
				 fill=tk.X, 
				 padx=5, 
				 pady=5)
		lab.pack(side=tk.LEFT)
		ent.pack(side=tk.RIGHT, 
				 expand=tk.YES, 
				 fill=tk.X)
		entries[field] = ent
	return entries

if gui == True:
	fields = ('Wickets_in_Hand', 'Overs_Remaining', 'Current_Target_Score', 'Number_of_neighbours_for_KNN','Year_of_ODI','First_Bat_Team','Second_Bat_Team','Home_ground_Second_Bat_Team_(1/0)', 'Month_of_ODI')
	if __name__ == '__main__':
		root = tk.Tk()
		root.title('CricPredic')
		root.geometry("1080x800+200+200")
		ents = makeform(root, fields)
		b1 = tk.Button(root, text='Run WINPredictor', bg = 'blue', command=(lambda e=ents: WINPredictor(e)))
		b1.pack(side=tk.LEFT, padx=5, pady=5)
		b3 = tk.Button(root, text='Quit', bg = 'blue', command=root.quit)
		b3.pack(side=tk.LEFT, padx=5, pady=5)
		root.mainloop()

