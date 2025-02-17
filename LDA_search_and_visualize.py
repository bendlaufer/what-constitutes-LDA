import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats as ss
import pandas as pd
import seaborn as sns
import random
import patsy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import sklearn.metrics as metrics
import polygon_basic
from scipy.stats import pearsonr

def get_results_lr(X_train,
                   X_train_male,
                   X_train_female,
                   y_train,
                   y_train_male,
                   y_train_female,
                   X_evaluate,
                   X_evaluate_male,
                   X_evaluate_female,
                   y_evaluate,
                   y_evaluate_male,
                   y_evaluate_female,
                   X_test,
                   X_test_male,
                   X_test_female,
                   y_test,
                   y_test_male,
                   y_test_female,
                   ml_model = 'LR'
                   , search_procedure = 'sample'
                   , data = 'germancredit'
                   , lam = 1
                   , n_seeds = 5000
                   , n_rows_max = 50000
                   , filename = 'result_data/result_df_germancredit_logistic.csv'
                   , append_info_to_name = True
                  ):
    
    columns = ['ml_model'
           ,'search_procedure'
           ,'data'
           ,'trial_seed'
           ,'train_utility'
           ,'train_disparity'
           ,'train_selection_rate'
           ,'evaluate_utility'
           ,'evaluate_disparity'
           ,'evaluate_selection_rate'
           ,'test_utility'
           ,'test_disparity'
           ,'test_selection_rate'
          ]
    result_df = pd.DataFrame(columns=columns)
    result_df['ml_model']=[-1]*50000
    
    for i in range(n_seeds):
        if i in [100*j for j in range(2000)]:
            print("seed ",i)
            
        #Search procedure
        if search_procedure=='sample':
            X_train_sample = pd.DataFrame(X_train).sample(frac=1,replace=True,random_state=i)
            y_train_sample = pd.DataFrame(y_train).sample(frac=1,replace=True,random_state=i)
            random_setting = 1
        elif search_procedure=='randomseed':
            X_train_sample = pd.DataFrame(X_train).copy()
            y_train_sample = pd.DataFrame(y_train).copy()
            random_setting = i
        else:
            print("ERROR: Unrecognized search procedure.")
            return
        
        #ML Model
        if ml_model=='LR':
            lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        elif ml_model=='RF':
            lr = RandomForestClassifier(max_depth=5
                                    ,random_state=random_setting#*max_depth
                                    ,class_weight="balanced"
                                   )
        elif ml_model=='DT':
            lr = tree.DecisionTreeClassifier(random_state=random_setting
                                             ,class_weight='balanced')
        else:
            print("ERROR: Unrecognized ML model.")
            return
        
        lr.fit(X_train_sample,np.array(y_train_sample).ravel())

        result_df.loc[i,'ml_model'] = ml_model
        result_df.loc[i,'search_procedure'] = search_procedure
        result_df.loc[i,'data'] = data
        result_df.loc[i,'trial_seed'] = i

        train_preds = lr.predict(X_train)
        train_preds_M = lr.predict(X_train_male)
        train_preds_F = lr.predict(X_train_female)

        evaluate_preds = lr.predict(X_evaluate)
        evaluate_preds_M = lr.predict(X_evaluate_male)
        evaluate_preds_F = lr.predict(X_evaluate_female)

        test_preds = lr.predict(X_test)
        test_preds_M = lr.predict(X_test_male)
        test_preds_F = lr.predict(X_test_female)

        result_df.loc[i,'train_utility'] = sum(train_preds*y_train.ravel())/sum(y_train.ravel()) - lam*(sum(train_preds*(1-y_train.ravel()))/sum(1-y_train.ravel()))
        result_df.loc[i,'evaluate_utility'] = sum(evaluate_preds*y_evaluate.ravel())/sum(y_evaluate.ravel()) - lam*(sum(evaluate_preds*(1-y_evaluate.ravel()))/sum(1-y_evaluate.ravel()))
        result_df.loc[i,'test_utility'] = sum(test_preds*y_test.ravel())/sum(y_test.ravel()) - lam*(sum(test_preds*(1-y_test.ravel()))/sum(1-y_test.ravel()))

        result_df.loc[i,'train_disparity'] = np.mean(train_preds_M)-np.mean(train_preds_F)
        result_df.loc[i,'evaluate_disparity'] = np.mean(evaluate_preds_M)-np.mean(evaluate_preds_F)
        result_df.loc[i,'test_disparity'] = np.mean(test_preds_M)-np.mean(test_preds_F)

        result_df.loc[i,'train_selection_rate'] = sum(train_preds)/len(train_preds)
        result_df.loc[i,'evaluate_selection_rate'] = sum(lr.predict(X_evaluate))/len(lr.predict(X_evaluate))
        result_df.loc[i,'test_selection_rate'] = sum(lr.predict(X_test))/len(lr.predict(X_test))
        
    result_df_toy = result_df.where(result_df['ml_model']!=-1).copy().dropna()
    
    if append_info_to_name:
        filename_final = filename[:-4] + "_info" + ml_model + search_procedure + data +'.csv'
    else:
        filename_final = filename
    result_df_toy.to_csv(filename_final)
    
    return result_df_toy

def get_analysis(result_df
                ,n_trials_min = 2
                ,n_trials_max = 100
                ,m_samples_per_n_trial=5000
                ,m_samples_for_perfectguess_bootstrap=250
                ,n_rows_max = 50000
                ,filename='result_data/analysis_df_germancredit_logistic.csv'
                ,append_info_to_name = True
               ):
    
    columns = ['ml_model'
           ,'search_procedure'
           ,'data'
           ,'n_trials_for_selection'
           ,'m_samples_per_n_trial'
           ,'avg_disparity_change'
           ,'UB_disparity_change'
           ,'LB_disparity_change'
           ,'avg_utility_change'
           ,'UB_utility_change'
           ,'LB_utility_change'
           ,'avg_disparity_rank'
           ,'UB_disparity_rank'
           ,'LB_disparity_rank'
           ,'avg_disparity_perfectguessrate'
           ,'UB_disparity_perfectguessrate'
           ,'LB_disparity_perfectguessrate'
           ,'avg_utility_rank'
           ,'UB_utility_rank'
           ,'LB_utility_rank'
          ]
    analysis_df = pd.DataFrame(columns = columns)
    analysis_df['ml_model']=[-1]*n_rows_max
    
    for i in range(n_trials_min,n_trials_max+1):
        if i in [20*j for j in range(2000)]:
            print("Size of set: ",i)
        disparity_change_list = []
        utility_change_list = []
        disparity_rank_list = []
        disparity_perfectguess_list = []
        utility_rank_list = []
        for j in range(m_samples_per_n_trial):
            rows_sample = result_df.sample(i).copy().reset_index(drop=True)
            best_evaluate_disparity_index = np.where(np.abs(rows_sample['evaluate_disparity'])==np.min(np.abs(rows_sample['evaluate_disparity'])))[0][0]
            disparity_change_list.append(np.abs(list(rows_sample['test_disparity'])[best_evaluate_disparity_index]) - np.mean(np.abs(rows_sample['test_disparity'])))
            utility_change_list.append(list(rows_sample['test_utility'])[best_evaluate_disparity_index] - np.mean(rows_sample['test_utility']))
            disparity_rank_list.append(list(ss.rankdata(np.abs(rows_sample['test_disparity'])))[best_evaluate_disparity_index])
            disparity_perfectguess_list.append(list(ss.rankdata(np.abs(rows_sample['test_disparity'])))[best_evaluate_disparity_index]==1)
            utility_rank_list.append(list(ss.rankdata(rows_sample['test_utility']))[best_evaluate_disparity_index])
        disparity_perfectguess_bootstraps = []
        for k in range(m_samples_per_n_trial):
            perfect_guess_sample = random.choices(disparity_perfectguess_list,k=len(disparity_perfectguess_list))
            disparity_perfectguess_bootstraps.append(np.mean(perfect_guess_sample))

        index = int(i-n_trials_min)
        analysis_df.loc[index,'ml_model'] = rows_sample.loc[0,'ml_model']
        analysis_df.loc[index,'search_procedure'] = rows_sample.loc[0,'search_procedure']
        analysis_df.loc[index,'data'] = rows_sample.loc[0,'data']
        analysis_df.loc[index,'n_trials_for_selection'] = i
        analysis_df.loc[index,'m_samples_per_n_trial'] = m_samples_per_n_trial
        analysis_df.loc[index,'avg_disparity_change']=np.mean(disparity_change_list)
        analysis_df.loc[index,'UB_disparity_change']=np.percentile(disparity_change_list,97.5)
        analysis_df.loc[index,'LB_disparity_change']=np.percentile(disparity_change_list,2.5)
        analysis_df.loc[index,'avg_utility_change']=np.mean(utility_change_list)
        analysis_df.loc[index,'UB_utility_change']=np.percentile(utility_change_list,97.5)
        analysis_df.loc[index,'LB_utility_change']=np.percentile(utility_change_list,2.5)
        analysis_df.loc[index,'avg_disparity_rank']=np.mean(disparity_rank_list)
        analysis_df.loc[index,'UB_disparity_rank']=np.percentile(disparity_rank_list,97.5)
        analysis_df.loc[index,'LB_disparity_rank']=np.percentile(disparity_rank_list,2.5)
        analysis_df.loc[index,'avg_disparity_perfectguessrate']=np.mean(disparity_perfectguess_list)
        analysis_df.loc[index,'UB_disparity_perfectguessrate']=np.percentile(disparity_perfectguess_bootstraps,97.5)
        analysis_df.loc[index,'LB_disparity_perfectguessrate']=np.percentile(disparity_perfectguess_bootstraps,2.5)
        analysis_df.loc[index,'avg_utility_rank']=np.mean(utility_rank_list)
        analysis_df.loc[index,'UB_utility_rank']=np.percentile(utility_rank_list,97.5)
        analysis_df.loc[index,'LB_utility_rank']=np.percentile(utility_rank_list,2.5)

    analysis_df_clean=analysis_df.where(result_df['ml_model']!=-1).copy().dropna()
    
    if append_info_to_name:
        filename_final = filename[:-4] + "_info" + str(rows_sample.loc[0,'ml_model']) + str(rows_sample.loc[0,'search_procedure']) + str(rows_sample.loc[0,'data']) +'.csv'
    else:
        filename_final = filename
    analysis_df_clean.to_csv(filename_final)
    
    return analysis_df_clean

def corrfunc(x, y, ax=None, **kws):
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.3f}', xy=(.1, .9), xycoords=ax.transAxes)

def visualize_pairplot(result_df,info_suffix):
    #Figure 1: Pairplot
    p = sns.pairplot(result_df, corner=True)#,size=10                )
    p.map_lower(corrfunc)
    plt.savefig('figures/pairplot_'+info_suffix+'.png',bbox_inches='tight',dpi=300)

def visualize_disputil(analysis_df,info_suffix):                     
    #Figure 2: Disp change
    plt.figure(figsize=[5,5])
    for k in range(len(analysis_df)):
        x=analysis_df.loc[k,'n_trials_for_selection']
        y1=analysis_df.loc[k,'UB_disparity_change']
        y2=analysis_df.loc[k,'LB_disparity_change']
        plt.plot([x,x],[y1,y2],color='black',linewidth=1)
        #plt.plot(analysis_df_clean['UB_disparity_change'])
        #plt.plot(analysis_df_clean['LB_disparity_change'])
    plt.plot([2,2],[0,0],color='black',label='95% CIs')
    sns.scatterplot(data= analysis_df, x='n_trials_for_selection',y='avg_disparity_change',color='black',linewidth=0,s=15,marker='o',label='Point estimate')
    plt.axhline(label='No effect')
    plt.title('Procedure impact on out-of-sample disparity with increasing draws')
    plt.legend()
    plt.savefig('figures/oos_disparity_'+info_suffix+'.png',bbox_inches='tight',dpi=300)
    
    #Figure 3: Disp perfectguess change
    plt.figure(figsize=[5,5])
    for k in range(len(analysis_df)):
        x=analysis_df.loc[k,'n_trials_for_selection']
        y1=analysis_df.loc[k,'UB_disparity_perfectguessrate']
        y2=analysis_df.loc[k,'LB_disparity_perfectguessrate']
        plt.plot([x,x],[y1,y2],color='black',linewidth=1)
        #plt.plot(analysis_df['UB_disparity_change'])
        #plt.plot(analysis_df['LB_disparity_change'])
    plt.plot([2,2],[0,0],color='black',label='95% CIs')
    sns.scatterplot(data= analysis_df, x='n_trials_for_selection',y='avg_disparity_perfectguessrate',color='black',linewidth=0,s=15,marker='o',label='Point estimate')
    plt.title('Observed frequency that in-sample optimal is out-of-sample optimal')
    plt.legend()
    plt.savefig('figures/oos_disparity_perfectguess_'+info_suffix+'.png',bbox_inches='tight',dpi=300)

    #Figure 4: Utility change
    plt.figure(figsize=[5,5])
    for k in range(len(analysis_df)):
        x=analysis_df.loc[k,'n_trials_for_selection']
        y1=analysis_df.loc[k,'UB_utility_change']
        y2=analysis_df.loc[k,'LB_utility_change']
        plt.plot([x,x],[y1,y2],color='black',linewidth=1)
        #plt.scatter(x,np.mean([y1,y2]))
        #plt.plot(analysis_df['UB_disparity_change'])
        #plt.plot(analysis_df['LB_disparity_change'])
    plt.plot([2,2],[0,0],color='black',label='95% CIs')
    sns.scatterplot(data= analysis_df, x='n_trials_for_selection',y='avg_utility_change',color='black',linewidth=0,s=20,marker='o',label='Point estimate')
    plt.axhline(label='No effect')
    plt.title('Procedure impact on out-of-sample utility with increasing draws')
    plt.legend()
    plt.savefig('figures/oos_utility_'+info_suffix+'.png',bbox_inches='tight',dpi=300)
    
    return
    