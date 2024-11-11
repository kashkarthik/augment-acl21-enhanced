# -*- coding: utf-8 -*-
"""
Generic scripts for managing input data for all classifiers

@author: Venelin
@author: Kaushik Karthikeyan
"""

import pandas as pd
import nltk
import numpy as np
import tensorflow as tf
import re
from datetime import datetime

# Mindreading general scripts and data used by all experiments

# Static variables
bio_cols = ['ID', 'Study', 'School', 'Class', 'Child', 'SEN (0 = No, 1 = Yes)',
            'EAL (0 = No, 1 = Yes)', 'DOB', 'DOT', 'Gender (0 = Girl, 1 = Boy)', 'MHVS']

text_cols = ['SFQuestion_1_Text', 'SFQuestion_2_Text', 'SFQuestion_3_Text', 
             'SFQuestion_4_Text', 'SFQuestion_5_Text', 'SFQuestion_6_Text', 
             'SS_Brian_Text', 'SS_Peabody_Text', 'SS_Prisoner_Text', 
             'SS_Simon_Text', 'SS_Burglar_Text']

rate_cols = ['SFQ1_Rating', 'SFQ2_Rating', 'SFQ3_Rating', 
             'SFQ4_Rating', 'SFQ5_Rating', 'SFQ6_Rating', 
             'SS_Brian_Rating', 'SS_Peabody_Rating', 'SS_Prisoner_Rating',
             'SS_Simon_Rating', 'SS_Burglar_Rating']

questions = ['Why did the men hide?', 'What does the woman think?', 'Why did the driver lock Harold in the van?', 
            'What is the deliveryman feeling and why?', 'Why did Harold pick up the cat?', 'Why does Harold fan Mildred?', 
            'Why does Brian say this?', 'Why did she say that?', 'Why did the prisoner say that?', 
            'Why will Jim look in the cupboard for the paddle?', 'Why did the burglar do that?']

# Function that reads all excel files and puts them in dataframes
def mr_excel_to_pd(fnames, calc_age=False):
    li = []
    for filename in fnames:
        try:
            df = pd.read_excel(filename, index_col=None, header=0)
            df.rename(columns={'DOB DD/MM/YYYY -99 = Missing' : 'DOB',
                               'EAL 0 = No, 1 = Yes' : 'EAL (0 = No, 1 = Yes)',
                               'Gender 0 = Girl, 1 = Boy' : 'Gender (0 = Girl, 1 = Boy)',
                               'SEN 0 = No, 1 = Yes' : 'SEN (0 = No, 1 = Yes)',
                               'SEN (0 = No, 1= Yes)' : 'SEN (0 = No, 1 = Yes)',
                               'SFQ1_Rating ': 'SFQ1_Rating'},
                      inplace=True)
            if calc_age:
                df['Age_Y'] = round((df['DOT'] - df['DOB']).dt.days/365)
            li.append(df)
        except:
            print(filename)
    
    full_frame = pd.concat(li, axis=0, ignore_index=True)
    return full_frame

def mr_create_qa(full_df, t_cols=text_cols, list_qs=questions):
    for text_col, question in zip(t_cols, list_qs):
        new_cname = text_col.replace("_Text", "_QT")
        full_df[new_cname] = question + full_df[text_col].astype(str)

def mr_tok_sc(response):
    tok_res = nltk.word_tokenize(response) if not pd.isna(response) else []
    return " ".join(tok_res)

def mr_create_datasets(full_df, X_cols, Y_cols):
    q_datasets = []
    for cur_X, cur_Y in zip(X_cols, Y_cols):
        cur_df = full_df[['ID', cur_X, cur_Y, 'Age_Y', 'Gender (0 = Girl, 1 = Boy)']].copy()
        cur_df[cur_X].replace('', np.nan, inplace=True)
        cur_df[cur_Y].replace(-99, np.nan, inplace=True)
        cur_df.dropna(inplace=True)
        cur_df["Question"] = cur_X
        cur_df.rename(columns={cur_X: 'Answer', cur_Y: 'Score', 'Age_Y': 'Age', 'ID': 'Child_ID', 'Gender (0 = Girl, 1 = Boy)': 'Gender'}, inplace=True)
        q_datasets.append([cur_X, cur_df])
    full_dataset = pd.concat([X for [_, X] in q_datasets])
    q_datasets.append(["full", full_dataset])
    return q_datasets

def mr_tf_data(var_X, var_y, BATCH_SIZE=4, SHUFFLE_BUFFER_SIZE=100):
    y_arr = var_y.to_numpy().astype(int)
    y_sm = tf.one_hot(y_arr, 3)
    var_dataset = tf.data.Dataset.from_tensor_slices((var_X, y_sm))
    var_dataset = var_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    return var_dataset

def mr_get_date(inp_date):
    year = re.search("20[0-9]{2}", inp_date)
    if year:
        return int(year.group(0))
    else:
        return 0

def mr_get_data(folder_name):
    out_l = []
    for cur_q in text_cols:
        inp_file = folder_name + cur_q + ".xlsx"
        cur_df = pd.read_excel(inp_file, index_col=None, header=0)
        out_l.append([cur_q, cur_df])
    full_dataset = pd.concat([X for [_, X] in out_l])
    out_l.append(["full", full_dataset])
    return out_l

def mr_get_qa_data(folder_name, t_cols=text_cols, list_qs=questions):
    out_l = []
    for cur_q, question in zip(t_cols, list_qs):
        inp_file = folder_name + cur_q + ".xlsx"
        cur_df = pd.read_excel(inp_file, index_col=None, header=0)
        cur_df['Answer'] = question + cur_df['Answer'].astype(str)
        out_l.append([cur_q, cur_df])
    full_dataset = pd.concat([X for [_, X] in out_l])
    out_l.append(["full", full_dataset])
    return out_l

# Function to load the augmented dataset
def load_augmented_data():
    # Path to the augmented data
    augmented_data_path = '../Data/augmented_dataset.csv'
    # Load the augmented dataset
    df = pd.read_csv(augmented_data_path)
    return df

# Function to combine the original and augmented datasets
def load_combined_data():
    original_data_path = '../Data/original_dataset.csv'
    augmented_data_path = '../Data/augmented_dataset.csv'
    df_original = pd.read_csv(original_data_path)
    df_augmented = pd.read_csv(augmented_data_path)
    combined_df = pd.concat([df_original, df_augmented], ignore_index=True)
    return combined_df
