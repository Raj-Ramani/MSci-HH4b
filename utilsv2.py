#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#pip install uproot awkward

# imports
from tqdm import tqdm
import uproot
import uproot3
import pandas as pd
import tensorflow as tf
from sklearn import tree
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


# In[ ]:


def load_nnt(
    file_path: str,
    trees: list = ["control", "validation", "sig"],
    columns: list = None,
    use_clusters: bool = True,
    flatten: bool = True,
) -> pd.DataFrame:
    """
    loads all the trees in the NNT into a single dataframe
    Note: This will not work with the triggerList tree.
    """
    list_of_dataframes = []

    for tree in trees:
        if use_clusters:
            clusters = list(uproot3.open(file_path)[tree].clusters())
            for cluster in tqdm(clusters):
                _df = uproot3.open(file_path)[tree].pandas.df(
                    columns,
                    entrystart=cluster[0],
                    entrystop=cluster[1],
                    flatten=flatten,
                )
                list_of_dataframes.append(_df)
        else:
            _df = uproot3.open(file_path)[tree].pandas.df(columns)
            list_of_dataframes.append(_df)

    df = pd.concat(list_of_dataframes)
    df = df.reset_index(drop=True)
    return df


def get_data(region,half="",mc=False,vr: bool = False):
    '''Open the files and import the data from years 2016, 2017 and 2018
    Calculates the NN_weights using the get_mu() function
    Inputs:
            file_path_data16,17,18: path of the data
            region: sig, validation or control
                    if None, import all three
            mc: if these are mc files, default False
    Outputs:
            data16,17,18: dataset of a specific year
    '''
    # if mc file, flatten is false
    if mc==True:
        flatten=False
    else:
        flatten=True
        
#     file_path_data_16 = []
#     file_path_data_17 = []
#     file_path_data_18 = []
#     for region in ["control", "validation", "sig"]:
#     with uproot.open("data16_NN_100_bootstraps.root:{}".format(region)) as file1:
#         data16 = pd.DataFrame(file1.arrays(file1.keys(), "(ntag != 3) & (X_wt_tag>=1.5)", library='np'))

#     with uproot.open("data17_NN_100_bootstraps.root:{}".format(region)) as file2:
#         data17 = pd.DataFrame(file2.arrays(file2.keys(), "(ntag != 3) & (X_wt_tag>=1.5)", library='np'))
  
    with uproot.open("data18_NN_100_bootstraps.root:{}".format(region)) as file3:
        data18 = pd.DataFrame(file3.arrays(file3.keys(), "(ntag != 3) & (X_wt_tag>=1.5)", library='np'))
        
    
        
       
        
#     file_path_data_16 = pd.concat(file_path_data_16, axis=0)
#     file_path_data_17 = pd.concat(file_path_data_17, axis=0)
#     file_path_data_18 = pd.concat(file_path_data_18, axis=0)

    # if None, import all three regions 
#     if region==None:
#         data16 = load_nnt(data16, flatten=flatten)
#         data17 = load_nnt(data17, flatten=flatten)
#         data18 = load_nnt(data18, flatten=flatten)
#     # specific region
#     else:
#         data16 = load_nnt(data16, trees=[region], flatten=flatten)
#         data17 = load_nnt(data17, trees= [region], flatten=flatten)
#         data18 = load_nnt(data18, trees = [region], flatten=flatten)

#     f_data16 = uproot.open("data16_NN_100_bootstraps.root")
#     f_data17 = uproot.open("data17_NN_100_bootstraps.root")
    f_data18 = uproot.open("data18_NN_100_bootstraps.root")
    
    # add year column
#     data16['year'] = 16
#     data17['year'] = 17
    data18['year'] = 18
    
    # add one-hot year column
#     data16['year_16'] = 1
#     data16['year_17'] = 0
#     data16['year_18'] = 0
#     data17['year_16'] = 0
#     data17['year_17'] = 1
#     data17['year_18'] = 0
    data18['year_16'] = 0
    data18['year_17'] = 0
    data18['year_18'] = 1
    
    # add one-hot bkt column
    bkt_vec = to_categorical(data18['trig_bucket'], num_classes=2)
    data18.insert(0, 'bkt_0', bkt_vec[:,0], False)
    data18.insert(0, 'bkt_1', bkt_vec[:,1], False)
    
    #initial masks
#     data16 = data16.loc[data16['X_wt_tag']>=1.5].reset_index(drop=True)
#     data17 = data17.loc[data17['X_wt_tag']>=1.5].reset_index(drop=True)
#     data18 = data18.loc[data18['X_wt_tag']>=1.5].reset_index(drop=True)

#     data16 = data16[~data16['pass_vbf_sel']].reset_index(drop=True)
#     data17 = data17[~data17['pass_vbf_sel']].reset_index(drop=True)
    data18 = data18[~data18['pass_vbf_sel']].reset_index(drop=True)
    
    if half == "even":
#         data16 = data16.loc[data16['event_number'] % 2 == 0]
#         data17 = data17.loc[data17['event_number'] % 2 == 0]
        data18 = data18.loc[data18['event_number'] % 2 == 0]
    elif half == "odd":
#         data16 = data16.loc[data16['event_number'] % 2 == 1]
#         data17 = data17.loc[data17['event_number'] % 2 == 1]
        data18 = data18.loc[data18['event_number'] % 2 == 1]
    
    # add NN_weights column if mc=False
    if mc==False:  
        if vr==True:
            norm_18_VR = get_mu(f_data18,18,vr=True)
            data18['NN_weights'] = norm_18_VR * data18['NN_d24_weight_VRderiv_bstrap_med_18']
            
            
        else:
            # calculate norm
    #         norm_16 = get_mu(f_data16, 16)
    #         norm_17 = get_mu(f_data17, 17)
            norm_18 = get_mu(f_data18, 18)

    #         data16['NN_weights'] = norm_16 * data16['NN_d24_weight_bstrap_med_16']
    #         data17['NN_weights'] = norm_17 * data17['NN_d24_weight_bstrap_med_17']
            data18['NN_weights'] = norm_18 * data18['NN_d24_weight_bstrap_med_18']
    
    return data18

def get_data_mask(data18,mask='2bRW'):
    '''apply mask to the data
    also add sample_weight and class columns
    note: this function could be appended to get_data(), but we may want to have 
    different masks for the same files and not load the files everytime,
    e.g. 2bRW and 4b masks in control region
    inputs:
            data16,17,18: outputs of the get_data() function
            mask: 2bRW or 4b, default=2bRW
    outputs:
            df: dataset after specific masks and concatenate all three years data
    '''
    # concatenate data
    data_all = data18
    if mask=='2bRW':
        df = data_all.loc[(data_all["ntag"] == 2) & (data_all["rw_to_4b"] == True)].reset_index(drop=True) 
        # background weights and class
        df['sample_weight'] = df['NN_weights']
        df['class'] = 0
    if mask=='4b':
        df = data_all.loc[data_all['ntag']>=4 & (data_all["rw_to_4b"] == True)].reset_index(drop=True)
        # signal weights and class
        df['sample_weight'] = 1
        df['class'] = 1
        
    return df

def get_mu(file, year: int = 16, vr: bool = False) -> float:
    """get nominal norm value from NNT"""
    vr_fix = "_VRderiv" if vr else ""
    return file[f"NN_norm{vr_fix}_bstrap_med_{year}"].member("fVal")

def build_model(hp):
    '''Deep neural network model used as input in the KerasTuner
    used for hyperparameter tuning
    input: hp is the hyperparameter variable in KerasTuner
    output: turnable model
    '''
    # sequential model
    model = Sequential()
    # number of layers is tunable 
    # each layer contains a dense and a dropout
    for i in range(hp.Int("num_layers", 2, 5, default=3)):
        # add dense layer
        model.add(
            Dense(
                # number of units is tunable , from 50 to 500
                units=hp.Int("units_" + str(i), min_value=50, max_value=500, step=50),
                # activation function is tunable , default relu
                activation=hp.Choice('act_' + str(i), ['relu', 'tanh'], default='relu')
            )
        )
        # add dropout layer
        model.add(
            Dropout(
                # rate is tunable, 0.0 meaning no dropout
                hp.Choice("rate_" + str(i), [0.0, 0.1, 0.2, 0.4])
            )
        )
    # add output dense layer; activation function is tunable, default is softmax
    model.add(Dense(2, activation=hp.Choice('act_output', ['sigmoid', 'softmax'], default='softmax')))
    model.compile(
        # learning rate is tunable, default 0.001
        optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4], default=1e-3)),
        loss="categorical_crossentropy",
        # use F1_Score() class as metric
        metrics=[F1_Score()],
    )
    return model

class F1_Score(tf.keras.metrics.Metric):
    '''f1 score metric used in TensorFlow or KerasTuner
    '''
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = tf.keras.metrics.Precision(thresholds=0.5)
        self.recall_fn = tf.keras.metrics.Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # precision and recall
        p = self.precision_fn(y_true[:,1], y_pred[:,1])
        r = self.recall_fn(y_true[:,1], y_pred[:,1])
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        return self.f1

    def reset_state(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()
        self.f1.assign(0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




