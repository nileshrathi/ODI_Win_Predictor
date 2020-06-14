import numpy as np
import math

f = open("04_cricket_1999to2011.csv", "r")
data = f.readlines()
f.close()

data = [a.split("\n")[0] for a in data]
data.pop(0)

data = [a.split(",") for a in data]

team_list = []
for d in data:
    team_list.append(d[18])

for d in data:
    team_list.append(d[19])

print(set(team_list))

team_list = list(set(team_list))
print(len(team_list))

match_ids = []
for d in data:
    match_ids.append(d[0])

match_ids = list(set(match_ids))
print(len(match_ids))




input_feature_list = []


for match_id in match_ids:
    
    inning_total_run = 0
    print(match_id)
    for d in data:
        if(d[0] == match_id):
            if(d[2] == "1"):
                one_hot_vector = [0]*len(team_list)
                one_hot_vector2 = [0]*len(team_list)
                inning_total_run = int(d[6])
                one_hot_vector[team_list.index(d[18])] = 1
                one_hot_vector2[team_list.index(d[19])] = 1
                one_hot_vector.extend(one_hot_vector2)
                one_hot_vector.extend([50-int(d[3])])
                one_hot_vector.extend([math.pow(int(d[11]), 2)])
                if(d[18] == d[20]):
                    one_hot_vector.extend([1])
                else:
                    one_hot_vector.extend([0])
                if(d[18] == d[26]):
                    one_hot_vector.extend([1])
                else:
                    one_hot_vector.extend([0])
                one_hot_vector.append(int(d[7]))
                input_feature_list.append(one_hot_vector)
    
    
    
for match_id in match_ids:
    
    inning_total_run = 0
    print(match_id)
    for d in data:
        if(d[0] == match_id):
            if(d[2] == "1"):
                one_hot_vector = [0]*len(team_list)
                one_hot_vector2 = [0]*len(team_list)
                one_hot_vector[team_list.index(d[18])] = 1
                one_hot_vector2[team_list.index(d[19])] = 1
                one_hot_vector.extend(one_hot_vector2)
            
                one_hot_vector.extend([50])
                one_hot_vector.extend([100])
                if(d[18] == d[20]):
                    one_hot_vector.extend([1])
                else:
                    one_hot_vector.extend([0])
                if(d[18] == d[26]):
                    one_hot_vector.extend([1])
                else:
                    one_hot_vector.extend([0])
                one_hot_vector.append(int(d[6]))
                input_feature_list.append(one_hot_vector)
                break

                
    

input_feature_list1 = [np.array(i) for i in input_feature_list]
input_feature_list2 = np.array(input_feature_list1)

np.save("one_hot_vector_w2.npy", input_feature_list2)




