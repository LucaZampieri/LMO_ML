'''
This file has not been thoughfully commented since just used to produces images and to increase our understanding of the dataset. 
Basically the functions whose name begins with "compare_" are just to plot the different features of the dataset
'''

import csv
import numpy as np
import pandas as pd

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def jet_separation(data):
    jet0 = data[data['PRI_jet_num'] == 0]
    jet1 = data[data['PRI_jet_num'] == 1]
    jet2 = data[data['PRI_jet_num'] == 2]
    jet3 = data[data['PRI_jet_num'] == 3]
    tot = jet0.shape[0]+jet1.shape[0]+jet2.shape[0]+jet3.shape[0]
    
    print(('Data: {data}\n jet0: {j0} {prop_j0}%\n jet1: {j1} {prop_j1}%\n jet2: {j2} {prop_j2}%'+\
          '\n jet3: {j3} {prop_j3}%\ntotat= {tot}').format\
          (data=data.shape, j0=jet0.shape, j1=jet1.shape, j2=jet2.shape, j3=jet3.shape, tot=tot,\
           prop_j0=jet0.shape[0]/data.shape[0]*100,\
           prop_j1=jet1.shape[0]/data.shape[0]*100,\
           prop_j2=jet2.shape[0]/data.shape[0]*100,\
           prop_j3=jet3.shape[0]/data.shape[0]*100 ))
    
    return jet0, jet1, jet2, jet3


def proportions(train, y_train, num_jet):
    num_back = y_train[train['PRI_jet_num']==num_jet].Prediction.value_counts()[-1]
    num_signal = y_train[train['PRI_jet_num']== num_jet].Prediction.value_counts()[1]
    prop = num_signal/(num_back+num_signal)
    print('Proportions of signal for jet{jet} are: {prop}'.format(jet=num_jet, prop=prop))
    
def give_bins_column(df,column,bin_num):
    sorted_col = df.sort_values([column])[column]
    minBin = sorted_col.min()
    maxBin = sorted_col.max()
    bins = np.linspace(minBin, maxBin, bin_num)
    return pd.cut(sorted_col,bins)

def compare_jets_sns(data,jet0,jet1,jet2,jet3,data_name):
    #needed_subplots = (len(jet0.columns)-1) # the first column being the index
    
    bin_num = 100
    nan_cols_jet1 = ['PRI_jet_subleading_phi','PRI_jet_subleading_eta','PRI_jet_subleading_pt',\
                   'DER_lep_eta_centrality','DER_prodeta_jet_jet','DER_mass_jet_jet','DER_deltaeta_jet_jet']
    nan_cols_jet0 = nan_cols_jet1 + ['PRI_jet_leading_phi','PRI_jet_leading_pt','PRI_jet_leading_eta','PRI_jet_all_pt']
    
    fig1, ax1 = plt.subplots(2*len(data.columns[0:]),4,sharex='row',sharey='row')
    fig1.set_figheight(len(data.columns[2:-1])*30)
    fig1.set_figwidth(60)
    my_list = [jet0,jet1,jet2,jet3]
    
    for k,column in enumerate (data.columns[:]):
        for i,jet in enumerate(my_list):
            if (column!='PRI_jet_num'):
                if (i==0):
                    if not(column in nan_cols_jet0):
                        signal = jet[jet['Prediction']==1.0]
                        ax1[2*k,i].set_title('{} in jet{}, signal'.format(column,i),fontsize=20,fontweight="bold")
                        sns.countplot(give_bins_column(signal,column,bin_num), ax = ax1[2*k,i],palette='autumn')

                        background = jet[jet['Prediction']==-1.0]
                        ax1[2*k+1,i].set_title('{} in jet{}, background'.format(column,i),fontsize=20,fontweight="bold")
                        sns.countplot(give_bins_column(background,column,bin_num), ax = ax1[2*k+1,i],palette='winter')
                elif (i==1):
                    if not(column in nan_cols_jet1):
                        signal = jet[jet['Prediction']==1.0]
                        ax1[2*k,i].set_title('{} in jet{}, signal'.format(column,i),fontsize=20,fontweight="bold")
                        sns.countplot(give_bins_column(signal,column,bin_num), ax = ax1[2*k,i],palette='autumn')

                        background = jet[jet['Prediction']==-1.0]
                        ax1[2*k+1,i].set_title('{} in jet{}, background'.format(column,i),fontsize=20,fontweight="bold")
                        sns.countplot(give_bins_column(background,column,bin_num), ax = ax1[2*k+1,i],palette='winter')
                else:
                    signal = jet[jet['Prediction']==1.0]
                    ax1[2*k,i].set_title('{} in jet{}, signal'.format(column,i),fontsize=20,fontweight="bold")
                    sns.countplot(give_bins_column(signal,column,bin_num), ax = ax1[2*k,i],palette='autumn')

                    background = jet[jet['Prediction']==-1.0]
                    ax1[2*k+1,i].set_title('{} in jet{}, background'.format(column,i),fontsize=20,fontweight="bold")
                    sns.countplot(give_bins_column(background,column,bin_num), ax = ax1[2*k+1,i],palette='winter')
            

    fig1.savefig(data_name+' jet_comparison_sns.png', bbox_inches='tight')
    
    
def compare_jets_tot(data,jet0,jet1,jet2,jet3,data_name):
    #needed_subplots = (len(jet0.columns)-1) # the first column being the index
    fig1, ax1 = plt.subplots(len(data.columns[0:]),4,sharey='row')
    fig1.set_figheight(len(data.columns[2:-1])*10)
    fig1.set_figwidth(30)
    my_list = [jet0,jet1,jet2,jet3]
    
    for k,column in enumerate (data.columns[2:]):
        for i,jet in enumerate(my_list):
            ax1[k,i].set_title('{} in jet{}'.format(column,i),fontsize=20,fontweight="bold")
            background = jet[jet['Prediction']==-1.0]
            signal = jet[jet['Prediction']==1.0]
            ax1[k,i].scatter(background['index'],\
                             background.sort_values([column])[column],\
                             color='b',s=0.01)
            ax1[k,i].scatter(signal['index'],\
                             signal.sort_values([column])[column],\
                             color='g',s=0.01)
    fig1.savefig(data_name+' jet comparison.png', bbox_inches='tight')
    
    
def compare_jets_dist(data,jet0,jet1,jet2,jet3,data_name):
    #needed_subplots = (len(jet0.columns)-1) # the first column being the index
    
    nan_cols_jet1 = ['PRI_jet_subleading_phi','PRI_jet_subleading_eta','PRI_jet_subleading_pt',\
                   'DER_lep_eta_centrality','DER_prodeta_jet_jet','DER_mass_jet_jet','DER_deltaeta_jet_jet']
    nan_cols_jet0 = nan_cols_jet1 + ['PRI_jet_leading_phi','PRI_jet_leading_pt','PRI_jet_leading_eta','PRI_jet_all_pt']
    
    fig1, ax1 = plt.subplots(len(data.columns[0:]),4,sharex='row',sharey='row')
    fig1.set_figheight(len(data.columns[2:-1])*15)
    fig1.set_figwidth(60)
    my_list = [jet0,jet1,jet2,jet3]
    
    for k,column in enumerate (data.columns[:]):
        for i,jet in enumerate(my_list):
            ax1[k,i].set_title('{} in jet{}'.format(column,i),fontsize=20,fontweight="bold")
            signal = jet[jet['Prediction']==1.0]
            background = jet[jet['Prediction']==-1.0]
            
            if (column!='PRI_jet_num'):
                if (i==0):
                    if not(column in nan_cols_jet0):
                        sns.distplot(background.sort_values([column])[column].dropna() ,ax = ax1[k,i],label='background')
                        sns.distplot(signal.sort_values([column])[column].dropna() ,ax = ax1[k,i],label='signal')
                elif (i==1):
                    if not(column in nan_cols_jet1):
                        sns.distplot(background.sort_values([column])[column].dropna() ,ax = ax1[k,i],label='background')
                        sns.distplot(signal.sort_values([column])[column].dropna() ,ax = ax1[k,i],label='signal')
                else:  
                    sns.distplot(background.sort_values([column])[column].dropna() ,ax = ax1[k,i],label='background')
                    sns.distplot(signal.sort_values([column])[column].dropna() ,ax = ax1[k,i],label='signal')
                    
            ax1[k,i].legend()
            ax1[k,i].set_xlabel( 'Equally spaced bins, from min to max' ,fontsize=12).set_weight('bold')
            ax1[k,i].set_ylabel('distribution', fontsize=12).set_weight('bold')

    fig1.savefig(data_name+' jet_comparison_dist.png', bbox_inches='tight')
