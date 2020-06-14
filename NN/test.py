import tkinter as tk
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import random
import matplotlib.pyplot as plt
from keras.models import load_model
import math


def predict_dl(match_data):
    result = []
    params = [282.26965637,238.72761843,207.21250311,168.57067616,137.4521217,103.82302121,78.50016798,50.58485567,26.79472072,11.66314597,10.91451836]
    for m_data in match_data:
        if(zFunc(int(m_data[0]), int(m_data[1]), params, params[10]) == None):
            result.append(0)
        else:
            result.append(zFunc(int(m_data[0]), int(m_data[1]), params, params[10]))
    
    # print(result)
    return result

def zFunc(x, w, params, L):
    if(w==10):
        return (params[0] * (1 - math.exp((-L*x)/params[0])))
    if(w==9):
        return (params[1] * (1 - math.exp((-L*x)/params[1])))
    if(w==8):
        return (params[2] * (1 - math.exp((-L*x)/params[2])))
    if(w==7):
        return (params[3] * (1 - math.exp((-L*x)/params[3])))
    if(w==6):
        return (params[4] * (1 - math.exp((-L*x)/params[4])))
    if(w==5):
        return (params[5] * (1 - math.exp((-L*x)/params[5])))
    if(w==4):
        return (params[6] * (1 - math.exp((-L*x)/params[6])))
    if(w==3):
        return (params[7] * (1 - math.exp((-L*x)/params[7])))
    if(w==2):
        return (params[8] * (1 - math.exp((-L*x)/params[8])))
    if(w==1):
        return (params[9] * (1 - math.exp((-L*x)/params[9])))

def WINPredictor(entries):
    Wickets_in_Hand= (int(entries['Wickets_in_Hand'].get()) )
    Overs_Remaining= (int(entries['Overs_Remaining'].get()) )
    Target_Score= (int(entries['Current_Target_Score'].get()) )
    First_Bat_Team= (str(entries['First_Bat_Team'].get()) )
    Second_Bat_Team= (str(entries['Second_Bat_Team'].get()) )
    Home_ground_Second_Bat_Team= (int(entries['Home_ground_Second_Bat_Team_(1/0)'].get()))
    Toss_Winning_Team = (str(entries['Toss_Winning_Team'].get()) )
    
    #print(Wickets_in_Hand,Overs_Remaining,Target_Score,Number_of_neighbours_for_KNN,Year_of_ODI,First_Bat_Team,Second_Bat_Team,Home_ground_Second_Bat_Team)

    #create_input_vector
    team_list = ['Netherlands', 'England', 'Bangladesh', 'Afghanistan', 'South Africa', 'Ireland', 'Pakistan', 'Canada', 'Africa XI', 'New Zealand', 'Bermuda', 'United Arab Emirates', 'Australia', 'West Indies', 'Asia XI', 'Sri Lanka', 'United States of America', 'Zimbabwe', 'Namibia', 'Scotland', 'ICC World XI', 'Hong Kong', 'Kenya', 'India']
    print(len(team_list))
    one_hot_vector = [0]*len(team_list)
    one_hot_vector2 = [0]*len(team_list)
    one_hot_vector[team_list.index(Second_Bat_Team)] = 1
    one_hot_vector2[team_list.index(First_Bat_Team)] = 1
    one_hot_vector.extend(one_hot_vector2)
    one_hot_vector.append(Overs_Remaining)
    one_hot_vector.append(Wickets_in_Hand)
    if(Second_Bat_Team == Home_ground_Second_Bat_Team):
        one_hot_vector.append(1)
    else:
        one_hot_vector.append(0)
    if(Second_Bat_Team == Toss_Winning_Team):
        one_hot_vector.append(1)
    else:
        one_hot_vector.append(0)
    
    input_features = np.array([one_hot_vector])
    input_features_dl = [[one_hot_vector[48], one_hot_vector[49]]]

    #make prediction
    model = load_model("./models/model_256_256_64.nn")
    results = model.predict(input_features)
    results_dl = predict_dl(input_features_dl)
    print(results)
    print(results_dl)
    

    if(Wickets_in_Hand > 3 and Overs_Remaining > 9):
        if(Target_Score == 0):
            Target_Score = 1
        if(Wickets_in_Hand in [1,2]):
            prob = 20 + (((results[0] - Target_Score)/Target_Score)*100)
        elif(Wickets_in_Hand in [3]):
            prob = 30 + (((results[0] - Target_Score)/Target_Score)*100)
        elif(Wickets_in_Hand in [4,5]):
            prob = 35 + (((results[0] - Target_Score)/Target_Score)*100)
        elif(Wickets_in_Hand in [6,7]):
            prob = 50 + (((results[0] - Target_Score)/Target_Score)*100)
        elif(Wickets_in_Hand in [8]):
            prob = 65 + (((results[0] - Target_Score)/Target_Score)*100)
        elif(Wickets_in_Hand in [9,10]):
            prob = 70 + (((results[0] - Target_Score)/Target_Score)*100)
        if(prob >= 100):
            prob = 99.98
        elif(prob <= 0):
            prob = random.randint(1,5)
        
        
    else:
        if(Target_Score == 0):
            Target_Score = 1
        prob = 50 + (((results_dl[0] - Target_Score)/Target_Score)*100)
        if(prob > 100):
            prob = 99
        elif(prob <= 0):
            prob = random.randint(1,6)

    #PREDICTION
    root = tk.Tk()
    root.title("Win Prediction")
    T = tk.Text(root,width = 40, height=4, font=("Helvetica", 32))
    T.pack()
    T.insert(tk.END, str(prob) + "%")
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
        if(field=='First_Bat_Team'):
            ent.insert(0, "West Indies")
        if(field=='Second_Bat_Team'):
            ent.insert(0, "India")
        if(field=='Home_ground_Second_Bat_Team_(1/0)'):
            ent.insert(0, "1")
        if(field=='Toss_Winning_Team'):
            ent.insert(0, "India")
        
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
gui = True
if gui == True:
    fields = ('Wickets_in_Hand', 'Overs_Remaining', 'Current_Target_Score','First_Bat_Team','Second_Bat_Team','Home_ground_Second_Bat_Team_(1/0)','Toss_Winning_Team')
    if __name__ == '__main__':
        root = tk.Tk()
        root.title("CricPredic")
        root.geometry("1080x800+200+200")
        ents = makeform(root, fields)
        b1 = tk.Button(root, text='Run WINPredictor',
        command=(lambda e=ents: WINPredictor(e)))
        b1.pack(side=tk.LEFT, padx=5, pady=5)
        b3 = tk.Button(root, text='Quit', command=root.quit)
        b3.pack(side=tk.LEFT, padx=5, pady=5)
        root.mainloop()