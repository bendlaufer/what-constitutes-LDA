import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import sklearn.metrics as metrics
import random

def classifier_util(classifier, GY, lam, useaccuracy=False):
    TP = 0
    FP = 0
    disp = 0
    
    #if classifier[0,0]==1:
    TP = TP + GY[0,0]*classifier[0,0]
    disp = disp + classifier[0,0]*GY[0,0]/(GY[0,0]+GY[0,1])
    #if classifier[0,1]==1:
    FP = FP + GY[0,1]*classifier[0,1]
    disp = disp + classifier[0,1]*GY[0,1]/(GY[0,0]+GY[0,1])
    #if classifier[1,0]==1:
    TP = TP + GY[1,0]*classifier[1,0]
    disp = disp - classifier[1,0]*GY[1,0]/(GY[1,0]+GY[1,1])
    #if classifier[1,1]==1:
    FP = FP + GY[1,1]*classifier[1,1]
    disp = disp - classifier[1,1]*GY[1,1]/(GY[1,0]+GY[1,1])
        
    n_pos = GY[0,0]+GY[1,0]
    n_neg = GY[0,1]+GY[1,1]
    util = TP/n_pos - lam* FP/n_neg
    
    if useaccuracy:
        print("NOTE: Using accuracy instead of utility formation.")
        print("GY: ",GY)
        print("classifier: ",classifier)
        TN = 0
        TN = TN + GY[1,1]*(1-classifier[1,1])
        TN = TN + GY[0,1]*(1-classifier[0,1])
        print("TN: ",TN)
        util = (TP + TN)/(n_pos+n_neg)
    
    return disp,util

def plot_basic_polygon(n_1pos,n_1neg,n_2pos,n_2neg, lam=1, filename = 'random_classifiers_possible.png', save=True, useaccuracy=False):
    
    GY = np.array([[n_1pos,n_1neg],[n_2pos,n_2neg]])
    perfect_classifier = np.array([[0,1],[0,1]])
    
    n_1=n_1pos+n_1neg
    n_2=n_2pos+n_2neg
    n_pos = n_1pos+n_2pos
    n_neg = n_1neg+n_2neg

    BR_1=n_1pos/n_1 
    BR_2=n_2pos/n_2
    base_rate_diff = BR_1-BR_2

    all_m_util = (n_1pos/n_pos)-lam*(n_1neg/n_neg)
    all_f_util = (n_2pos/n_pos)-lam*(n_2neg/n_neg)
    
    fig,ax = plt.subplots(figsize=[4,4])
    
    sns.scatterplot([base_rate_diff],[1],marker="^",color='orange',label='Perfect accuracy',s=80,linewidth=0)
    sns.scatterplot([1],[all_m_util],marker=">",color='orange',label='Max disparity',s=80,linewidth=0)

    polygon_xs = np.array([base_rate_diff,0,-1,0,-base_rate_diff,0,1,0])
    polygon_ys = np.array([1,0,all_f_util,0,-lam,0,all_m_util,0])#polygon_ys = np.array([1,0,all_f_util,0,-1,0,all_m_util,0])

    #Two candidates:
    if n_1/n_pos < lam*n_2/n_neg:#n_1/n_pos < n_2/n_neg:
        classifier_1 = np.array([[0,0],[1,0]])
        x,y=classifier_util(classifier_1, GY, lam,useaccuracy=useaccuracy)
        classifier_1_inv = np.array([[1,1],[0,1]])
        x_inv,y_inv=classifier_util(classifier_1_inv, GY, lam,useaccuracy=useaccuracy)
        polygon_xs[1]=x
        polygon_xs[5]=x_inv
        polygon_ys[1]=y
        polygon_ys[5]=y_inv
        #ax = sns.scatterplot([x,x_inv],[y,y_inv],label='If G=1: 0, else: perfect')
        ax = sns.scatterplot([x],[y],label='If G=1: 0; else: perfect',marker='P',s=80,linewidth=0)
    else:
        classifier_2 = np.array([[1,0],[1,1]])
        x,y=classifier_util(classifier_2, GY, lam,useaccuracy=useaccuracy)
        classifier_2_inv = np.array([[0,1],[0,0]])
        x_inv,y_inv=classifier_util(classifier_2_inv, GY, lam,useaccuracy=useaccuracy)
        polygon_xs[1]=x
        polygon_xs[5]=x_inv
        polygon_ys[1]=y
        polygon_ys[5]=y_inv
        #ax = sns.scatterplot([x,x_inv],[y,y_inv],label='If G=1: perfect, else: 1')
        ax = sns.scatterplot([x],[y],label='If G=1: perfect; else: 1',marker='P',s=80,linewidth=0)
    if lam*n_1/n_neg < n_2/n_pos:#n_1/n_neg < n_2/n_pos:
        classifier_3 = np.array([[1,1],[1,0]])
        x,y=classifier_util(classifier_3, GY, lam,useaccuracy=useaccuracy)
        classifier_3_inv = np.array([[0,0],[0,1]])
        x_inv,y_inv=classifier_util(classifier_3_inv, GY, lam,useaccuracy=useaccuracy)
        polygon_xs[7]=x
        polygon_xs[3]=x_inv
        polygon_ys[7]=y
        polygon_ys[3]=y_inv
        #ax = sns.scatterplot([x,x_inv],[y,y_inv],label='If G=1: 1, else: perfect')
        ax = sns.scatterplot([x],[y],label='If G=1: 1; else: perfect',marker='X',s=80,linewidth=0)
    else:
        classifier_4 = np.array([[1,0],[0,0]])
        x,y=classifier_util(classifier_4, GY, lam,useaccuracy=useaccuracy)
        classifier_4_inv = np.array([[0,1],[1,1]])
        x_inv,y_inv=classifier_util(classifier_4_inv, GY, lam,useaccuracy=useaccuracy)
        polygon_xs[7]=x
        polygon_xs[3]=x_inv
        polygon_ys[7]=y
        polygon_ys[3]=y_inv
        #ax = sns.scatterplot([x,x_inv],[y,y_inv],label='If G=1: perfect, else: 0')
        ax = sns.scatterplot([x],[y],label='If G=1: perfect; else: 0',marker='X',s=80,linewidth=0)

    #Plot the highest-performance-achievable fair classifier
    if base_rate_diff>0 and n_1/n_pos < lam*n_2/n_neg:#n_2/n_neg:
        best_fair_util = 1-np.abs(base_rate_diff*n_1/n_pos)
    elif base_rate_diff>0 and n_1/n_pos >= lam*n_2/n_neg:#n_2/n_neg:
        best_fair_util = 1-np.abs(base_rate_diff*lam*n_2/n_neg)#n_2/n_neg)
    elif base_rate_diff<=0 and lam*n_1/n_neg < n_2/n_pos:#n_1/n_neg < n_2/n_pos:
        best_fair_util = 1-np.abs(base_rate_diff*lam*n_1/n_neg)#n_1/n_neg)
    else:
        best_fair_util = 1-np.abs(base_rate_diff*n_2/n_pos)
    sns.scatterplot([0],[best_fair_util],label="Optimal 0-disparity classifier",zorder=10,s=50,linewidth=0)    

    ax.fill(polygon_xs,polygon_ys,alpha=0.4,label='Feasible random classifiers',color='#1f77b4',linewidth=0,zorder=-1)

    ax.set_ylabel("Utility",loc='bottom')
    ax.set_xlabel("Disparity",loc='left')

    #ax.set_title("Feasible utility and disparity")

    ax.set_xticks([-1,1])#-.5,.5,1])
    ax.set_xlim([-1.05,1.05])
    ax.set_yticks([-1,1])#.5,-.5,1])
    ax.set_ylim([-1.05,1.05])
    #ax.set_ylim([-10.05,1.05])

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    ""# Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.legend(loc='center left', bbox_to_anchor=(.55,.185))#(1, 0.5))

    if save:
        plt.savefig(filename,bbox_inches='tight',dpi=300)

    return polygon_xs, polygon_ys
