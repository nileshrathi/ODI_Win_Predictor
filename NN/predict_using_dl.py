import numpy as np
import random
import matplotlib.pyplot as plt
import math

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

def plot(match_data, results):
    xdata = []
    ydata = []
    annot = []
    lenth = len(match_data) - 1
    print(len(match_data))
    while(lenth >=0 ):
        xdata.append(match_data[lenth][2])
        ydata.append(match_data[lenth][1])
        annot.append(str(match_data[lenth][0]) + ', ' + str(results[lenth]))
        lenth = lenth - 1
        
    fig, ax = plt.subplots()
    ax.scatter(xdata, ydata)
    for i, txt in enumerate(annot):
        annot = ax.annotate(txt, (xdata[i], ydata[i]))
    #mplcursors.cursor(hover = True)
    plt.show()
    fig.savefig('plot.png')

match_data1 = np.load("match_536931.npy")
print(len(match_data1))
match_data_true_label = match_data1[:,[52]].copy()
match_data = match_data1[:,0:52].copy()
print("Geragr" + str(len(match_data_true_label)))

def predict_dl(match_data):
    result = []
    params = [282.26965637,238.72761843,207.21250311,168.57067616,137.4521217,103.82302121,78.50016798,50.58485567,26.79472072,11.66314597,10.91451836]
    for m_data in match_data:
        if(zFunc(int(m_data[48]), int(m_data[49]), params, params[10]) == None):
            result.append(0)
        else:
            result.append(zFunc(int(m_data[48]), int(m_data[49]), params, params[10]))
    
    # print(result)
    return result
        

prob_list = []
results = predict_dl(match_data)
for i in range(len(results)):
    prob = 50 + (((results[i] - match_data_true_label[i])/match_data_true_label[i])*100)
    if(prob[0] > 100):
        prob[0] = 99
    elif(prob[0] <= 0):
        prob[0] = random.randint(0,5)
    prob_list.append(prob[0])

print(prob_list)  
# for i in range(len(X_test)):
#     print(str(X_test[i]) + " : " + str(y_test[i]) + " : " + str(results[i]))

#plot(match_data, results)
# plt.ylim(ymin=0)
# plt.ylim(ymax=100)
# plt.scatter(range(0,19), prob_list)
# plt.show()

