# -*- coding: utf-8 -*-
"""
@author: Venelin Kovatchev
@author: Kaushik Karthikeyan


The class for the BILSTM classifier

The class uses TensorFlow BILSTM as a core model

Different methods take care of processing the data in a standardized way

"""

import pandas as pd
import numpy as np
import scipy
import nltk
import spacy
import gensim
import glob
import csv
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import cross_val_score
import sklearn.model_selection
import sklearn.pipeline
import re
from sklearn import svm
from sklearn import *
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
import fasttext
from scipy import sparse
import tensorflow_datasets as tfds
import tensorflow as tf
import collections
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import LeaveOneOut,KFold,train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score



# Custom imports

from mr_generic_scripts import *
from mr_generic_scripts import load_combined_data

class MR_bilstm:
    
    def __init__(self, text_cols, age_list, v_size, max_len):
        
        # Initialize the core variables
        
        # The current classifier 
        self.mr_c = None
        
        # The current tokenizer
        self.mr_tok = None
        
        # Initialize model variables
        
        self.mr_set_model_vars(text_cols, age_list, v_size, max_len)
    
    # Function that sets model variables
    # Input: list of questions, list of ages, size of vocabulary, max len of sentence
    # Also includes certain pre-build variables for truncating
    # Also includes certain pre-built variables for dataset creation (batch size, shuffle buffer)
    def mr_set_model_vars(self, text_cols, age_list, v_size, max_len, 
                          trunc_type = 'post', padding_type = 'post', oov_tok = '<OOV>',
                          batch_size = 4, shuffle_buffer_size = 100):
        
        # List of questions
        self.q_list = text_cols
        
        # List of ages
        self.age_list = age_list
        
        # Size of the vocabulary
        self.v_size = v_size
        
        # Padding length
        self.max_len = max_len
        
        # Truncating type
        self.trunc_type = trunc_type
        
        # Padding type
        self.padding_type = padding_type
        
        # Token to replace OOV tokens
        self.oov_tok = oov_tok
        
        # Batch size for tf_dataset
        self.batch_size = batch_size
        
        # Shuffle buffer size
        self.shuffle_buffer_size = shuffle_buffer_size
        

    # Function that sets evaluation variables
    def mr_set_eval_vars(self, eval_q, eval_age, return_err = False):
        
        # Whether or not to perform evaluation by question
        self.eval_q = eval_q
        
        # Whether or not to perform evaluation by age
        self.eval_age = eval_age
        
        # Whether or not to return wrong predictions
        self.return_err = return_err

    # Convert the text from words to indexes and pad to a fixed length (needed for LSTM)
    # Input - text
    # Uses model variables for vocabulary size, token to be used for OOV, padding and truncating
    def mr_t2f(self, inp_text):
        
        # Check if a tokenizer already exists
        # If it is None, this is the first time we run the function -> fit the tokenizer
        if self.mr_tok == None:
            # Initialize the tokenizer
            self.mr_tok = Tokenizer(num_words = self.v_size, oov_token=self.oov_tok)
            
            # Fit the tokenizer
            self.mr_tok.fit_on_texts(inp_text)
            
        # Convert the dataset
        indexed_dataset = self.mr_tok.texts_to_sequences(inp_text)
        
        # Pad to max length
        padded_dataset = pad_sequences(indexed_dataset, 
                                       maxlen = self.max_len, 
                                       padding = self.padding_type, 
                                       truncating = self.trunc_type)
        
        # Return the converted dataset
        return padded_dataset
    
    # Function that created a tensorflow dataset from X and Y
    # Input: X and Y
    def mr_tf_data(self, var_X, var_y):
        
        # Convert the labels in proper format
        y_arr = var_y.to_numpy().astype(int)
        
        # Create the actual dataset and shuffle it        
        var_dataset = tf.data.Dataset.from_tensor_slices((var_X, y_arr))  
        var_dataset = var_dataset.shuffle(self.shuffle_buffer_size).batch(self.batch_size)
    
        return var_dataset
    
    # Function that converts a dataframe to a dataset
    # Input - dataframe
    def mr_to_dataset(self, cur_df):
        # X is the answer column
        cur_X = cur_df["Answer"]
        # Y is the score column
        cur_Y = cur_df["Score"]
        
        # Convert X to a one-hot vector representation
        # The vector is of a predefined fixed length and uses a fixed vocabulary size
        X_idx = self.mr_t2f(cur_X)
        
        # Create the dataset
        cur_dataset = self.mr_tf_data(X_idx,cur_Y)
        
        # Return everything
        return(X_idx, cur_Y, cur_dataset)
    
    # Function that trains the classifier
    # Input - a train set, and a val set
    def mr_train(self, train_df, val_df):
        # Reset the tokenizer and the model at the start of each training
        self.mr_c = None
        self.mr_tok = None       

        # Load combined data for training
        train_df = load_combined_data()  # Load the combined (original + augmented) dataset

        # Convert dataframes to datasets
        X_train_idx, y_train, train_dataset = self.mr_to_dataset(train_df)
        X_val_idx, y_val, val_dataset = self.mr_to_dataset(val_df)
        
        train_dataset = train_dataset.repeat() 
        val_dataset = val_dataset.repeat()
        
        print(f"Train DataFrame size: {len(train_df)}")
        print(f"Validation DataFrame size: {len(val_df)}")
        
        # print(f"Shape of X_train_idx: {np.shape(X_train_idx)}")
        # print(f"Shape of y_train: {np.shape(y_train)}")
        
        steps_per_epoch = 1
        validation_steps = 1


        # Current shape var
        inp_shape = np.shape(X_train_idx[0])[0]
        self.input_shape = inp_shape 
        print(f"Input shape: {inp_shape}") 
        

        for data in train_dataset.take(1):
            inferred_batch_size = data[0].shape[0]
            print(f"Inferred batch size from train_dataset: {inferred_batch_size}")

        # steps_per_epoch = max(1, (len(train_df) // inferred_batch_size) -1)

        # Define a vanilla BILSTM model
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(int(np.shape(X_train_idx[0])[0]),)),
            # Word embedding layers, size of the vocabulary X 64 dimensions
            tf.keras.layers.Embedding(1000, 64),
            # BILSTM layer, same dimensions as embeddings
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            # Dense relu layer on top of the BILSTM
            tf.keras.layers.Dense(64, activation='relu'),
            # Add dropout to reduce overfitting
            tf.keras.layers.Dropout(.5),
            # Softmax classification for 3 classes
            tf.keras.layers.Dense(3,activation='softmax')
        ])

        # Compile the model
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(1e-4),
            metrics=['accuracy']
        )        

        # Print the model summary
        print(model.summary())

        print('\n Training')

        # Train
        
        history = model.fit(
            train_dataset,
            epochs=20,
            validation_data=val_dataset,
            # validation_steps=30,
            steps_per_epoch=steps_per_epoch,# Set custom steps_per_epoch
            validation_steps=validation_steps,
            verbose=0 
        )
        # history = model.fit(train_dataset, epochs=20,
        #                     validation_data=val_dataset, 
        #                     validation_steps=30)

        # Update the current model variable
        self.mr_c = model

    
    # def mr_train(self, train_df, val_df):
        
    #     # Reset the tokenizer and the model at the start of each training
        
    #     self.mr_c = None
    #     self.mr_tok = None       
        
    #     # Convert dataframes to datasets
    #     X_train_idx, y_train, train_dataset = self.mr_to_dataset(train_df)
    #     X_val_idx, y_val, val_dataset = self.mr_to_dataset(val_df)
        
    #     # Current shape var
    #     inp_shape = np.shape(X_train_idx[0])[0]
    
    #     # Define a vanilla BILSTM model
    #     model = tf.keras.Sequential([
    #         # Input layer
    #         tf.keras.layers.Input(shape=(inp_shape)),
    #         # Word embedding layers, size of the vocabulary X 64 dimensions
    #         tf.keras.layers.Embedding(1000, 64),
    #         # BILSTM layer, same dimensions as embeddings
    #         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    #         # Dense relu layer on top of the BILSTM
    #         tf.keras.layers.Dense(64, activation='relu'),
    #         # Add dropout to reduce overfitting
    #         tf.keras.layers.Dropout(.5),
    #         # Softmax classification for 3 classes
    #         tf.keras.layers.Dense(3,activation='softmax')
    #     ])
        
    #     # Compile the model
    #     model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #                   optimizer=tf.keras.optimizers.Adam(1e-4),
    #                   metrics=['accuracy'])        
        
    #     # Print the moodel setting
    #     print(model.summary())
        
    #     print('\n Training')
        
    #     # Train
    #     history = model.fit(train_dataset, epochs=20,
    #                         validation_data=val_dataset, 
    #                         validation_steps=30)
            
    #     # Update the current model variable
    #     self.mr_c = model
        
    # Function that evaluates the model on a test set
    # Input - test set
    
    def mr_test(self, test_df):
        # Check available columns in test_df
        print("Columns in test_df:", test_df.columns)

        # Assuming 'Score' is the target column and 'Question' is the input text column
        y_test = test_df["Score"].values
        # Tokenize the input text data in 'Question' column
        X_test_idx = self.mr_tok.texts_to_sequences(test_df["Question"])
        # # Pad sequences to ensure consistent input shape
        X_test_idx = tf.keras.preprocessing.sequence.pad_sequences(X_test_idx, maxlen=self.input_shape, padding='post')
        # # Predict using the trained model
        y_pred = self.mr_c.predict(X_test_idx)
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred.argmax(axis=1))  # Use argmax to get the class predictions
        test_f1 = f1_score(y_test, y_pred.argmax(axis=1), average="macro")
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Test Macro F1: {test_f1}")
        return test_accuracy, test_f1


    # def mr_test(self, test_df):
        
    #     # Initialize output vars
    #     acc_scores = []
    #     f1_scores = []
        
    #     # Convert the dataframe to a dataset
    #     X_test_idx, y_test, test_dataset = self.mr_to_dataset(test_df)
        
    #     print("Testing the model on the test set:")
        
    #     # Run the model internal evaluation on the test set
    #     test_loss, test_acc = self.mr_c.evaluate(test_dataset)
        
    #     # Get the actual predictions of the model for the test set
    #     #y_pred = self.mr_c.predict_classes(X_test_idx)
    #     y_pred = np.argmax(self.mr_c.predict(X_test_idx), axis=-1)
        
    #     print("y_test (actual values):", y_test.tolist())  # Check target values
    #     print("y_pred (predicted values):", y_pred.tolist())  # Check predicted values
    #     print("Evaluating test set macro F1 score...")
    #     # Calculate macro F1
    #     macro_score = sklearn.metrics.f1_score(y_test.tolist(), 
    #                                            [float(ele) for ele in y_pred],
    #                                            average='macro')
        
    #     print('Test Macro F1: {} \n'.format(round(macro_score,2)))
        
    #     # Add the results to the output
    #     acc_scores.append(round(test_acc,2))
    #     f1_scores.append(round(macro_score,2))
        
    #     # Test by question (if requested)
    #     # Add the scores to the output
    #     # Otherwise add empty list
    #     if self.eval_q:
    #         qa_scores, qf_scores = self.mr_eval_col(test_df,"Question",self.q_list)
            
    #         acc_scores.append(qa_scores)
    #         f1_scores.append(qf_scores)
    #     else:
    #         acc_scores.append([])
    #         f1_scores.append([])
            
    #     # Test by age (if requested)
    #     # Add the scores to the output
    #     # Otherwise add empty list    
    #     if self.eval_age:
    #         aa_scores, af_scores = self.mr_eval_col(test_df,"Age",self.age_list)
            
    #         acc_scores.append(aa_scores)
    #         f1_scores.append(af_scores)
    #     else:
    #         acc_scores.append([])
    #         f1_scores.append([])
            
    #     return(acc_scores,f1_scores)
            
            
    # Function that evaluates the model by a specific column
    # Can also return the actual wrong predictions
    # Input - test set, column, values
    def mr_eval_col(self, test_df, col_name, col_vals):
        # Initialize output
        acc_scores = []
        f1_scores = []
        
        # Initialize output for wrong predictions, if needed
        if self.return_err:
            wrong_pred = []
        
        # Loop through all values
        for col_val in col_vals:
            print(f"Evaluating column: {col_name}, value: {col_val}")
            # Initialize output for wrong predictions, if needed
            if self.return_err:
                cur_wrong = []
            
            # Get only the entries for the current value
            cur_q = test_df[test_df[col_name] == col_val].copy()
            
            # Convert dataframe to dataset
            X_test_idx, y_test, test_dataset = self.mr_to_dataset(cur_q)
            
            print("y_test (actual values):", y_test.tolist())
            print("Evaluating column-based macro F1 score...")
            
            # print("Evaluating column {} with value {}".format(col_name,col_val))
            
            # Print the internal evaluation
            test_loss, test_acc = self.mr_c.evaluate(test_dataset)
            
            # Get the actual predictions of the model for the test set
            #y_pred = self.mr_c.predict_classes(X_test_idx)
            y_pred = np.argmax(self.mr_c.predict(X_test_idx), axis=-1)
            print("y_pred (predicted values):", y_pred.tolist())
            # Calculate macro F1
            macro_score = sklearn.metrics.f1_score(y_test.tolist(), 
                                                   [float(ele) for ele in y_pred],
                                                   average='macro')
            
            print('Macro F1: {} \n'.format(round(macro_score,2)))    
            
            # Add the results to the output
            acc_scores.append(round(test_acc,2))
            f1_scores.append(round(macro_score,2))
            
            if self.return_err:
                # Loop through all predictions and keep the incorrect ones
                # cur_q["Answer"], y_test, and y_pred are all matched, since they
                # are not shuffled (shuffle only applies to the test_dataset)
                for c_text,c_gold,c_pred in zip(cur_q["Answer"],y_test.tolist(),
                                                [float(ele) for ele in y_pred]):
                    if c_pred != c_gold:
                        cur_wrong.append([c_text,c_gold,c_pred])
                wrong_pred.append(cur_wrong)
            
        # Return the output
        if self.return_err:
            return(acc_scores,f1_scores, wrong_pred)
        else:
            return(acc_scores, f1_scores)
      
    
    # Function for a dummy one run on train-test
    # Input - full df, ratio for splitting on train/val/test, return errors or not
    
    def mr_one_train_test(self, full_df, test_r, val_r=0):
        # Load combined data
        full_df = load_combined_data()

        # Split train and test
        train_df, test_df = train_test_split(full_df, test_size=test_r)

        # Check if we also need val
        if val_r > 0:
            train_df, val_df = train_test_split(train_df, test_size=val_r)
        else:
            val_df = test_df

        # Train the classifier
        self.mr_train(train_df, val_df)

        # Test the classifier
        return self.mr_test(test_df)

    
    # def mr_one_train_test(self, full_df, test_r, val_r=0):
        
    #     # Split train and test
    #     train_df, test_df = train_test_split(full_df, test_size = test_r)
        
    #     # Check if we also need val
    #     if val_r > 0:
    #         train_df, val_df = train_test_split(train_df, test_size = val_r)
    #     else:
    #         # If not, validation is same as test
    #         val_df = test_df
            
    #     # Train the classifier
    #     self.mr_train(train_df, val_df)
        
    #     # Test the classifier
    #     return(self.mr_test(test_df))
    
    
    # Function for a dummy one-run on a provided train-test split
    # Input - train_df, test_df, ratio for splitting val
    def mr_one_run_pre_split(self,train_df, test_df, val_r = 0):
        # Check if we also need val
        if val_r > 0:
            train_df, val_df = train_test_split(train_df, test_size = val_r)
        else:
            # If not, validation is same as test
            val_df = test_df        
            
        # Train the classifier
        self.mr_train(train_df, val_df)
        
        # Test the classifier
        return(self.mr_test(test_df))

    # Function for a dummy one-vs-all runs training on 10 questions and evaluating on the 11th
    # Input - full df, ratio for splitting val
    def mr_one_vs_all_q(self, full_df, val_r=0.25):
        # Initialize output
        acc_scores = []
        f1_scores = []
        # Loop over all the questions
        for cur_q in range(11):
            # Drop the full dataset, we don't need it for the current experiment
            # Get only the dataframes
            cur_train = [x[1] for x in full_df[:11]]
            # Get the current id question for testing, rest is for training
            cur_test = cur_train.pop(cur_q)
            
            cur_train = pd.concat(cur_train)
            
            # Debug info
            print("Training one vs all for question {}; Train size: {}, Test size: {}".
                  format(full_df[cur_q][0],len(cur_train),len(cur_test)))
            
            # Train and test
            cur_acc, cur_f1 = self.mr_one_run_pre_split(cur_train, cur_test, val_r)
            
            
            # Keep the results
            acc_scores.append(cur_acc[0])
            f1_scores.append(cur_f1[0])
            
        return(acc_scores, f1_scores)
    
    # Function for a dummy multiple runs (random splits, no 10-fold)
    # Input - full df, ratio for splitting on train/val/test, number of runs
    def mr_multi_train_test(self, full_df, test_r, val_r=0, num_runs=10):
        
        # Initialize output
        all_results = []
        
        # Run a classifier num_runs times
        for cur_run in range(num_runs):
            cur_acc, cur_f1 = self.mr_one_train_test(full_df, test_r, val_r)
            
            all_results.append((cur_acc, cur_f1))
            
        return(all_results)
    
      
    #Function for a 10-fold cross validation
    # Input - full df, ratio for splitting on train/val/test, number of runs
    def mr_kfold_train_test(self, full_df, val_r=0.25, num_runs=10, r_state = 42):
        
        # Initialize output
        all_results = []        
        
        # Run k-fold split
        kf = KFold(n_splits=num_runs, shuffle=True, random_state = r_state)
        
        # Run different splits
        for train_index, test_index in kf.split(full_df):
            train_df = full_df.iloc[train_index]
            test_df = full_df.iloc[test_index]
            cur_acc, cur_f1 = self.mr_one_run_pre_split(train_df, test_df, val_r)
            
            all_results.append((cur_acc, cur_f1))
            
        return(all_results)
    
    # Function for a dummy 10-fold cross validation with a predefined test set
    # Input - full df, test df, ratio for splitting on val, number of runs
    
    def mr_kfold_pre_split(self, full_df, test_df, val_r=0.25, num_runs=10, r_state=42):
        # Load the combined dataset
        full_df = load_combined_data()  # Load combined data for training

        # Initialize output
        all_results = []        

        # Run k-fold split
        kf = KFold(n_splits=num_runs, shuffle=True, random_state=r_state)

        # Run different splits
        for train_index, test_index in kf.split(full_df):
            train_df = full_df.iloc[train_index]
            kv_test_df = full_df.iloc[test_index]
            
            print(f"Fold Train Size: {len(train_df)}, Fold Validation Size: {len(kv_test_df)}, Test Size: {len(test_df)}")

        # Check if the split datasets are empty
            if train_df.empty or kv_test_df.empty:
                raise ValueError("One of the KFold splits resulted in an empty DataFrame.")

            # Train on the cv, same as normal
            kv_cur_acc, kv_cur_f1 = self.mr_one_run_pre_split(train_df, kv_test_df, val_r)

            # Extra evaluation on the predefined test
            cur_acc, cur_f1 = self.mr_test(test_df)

            # Return all the results
            all_results.append((kv_cur_acc, kv_cur_f1, cur_acc, cur_f1))

        return all_results

    
    # def mr_kfold_pre_split(self, full_df, test_df, val_r=0.25, num_runs=10, r_state = 42):
        
    #     # Initialize output
    #     all_results = []        
        
    #     # Run k-fold split
    #     kf = KFold(n_splits=num_runs, shuffle=True, random_state = r_state)
        
    #     # Run different splits
    #     for train_index, test_index in kf.split(full_df):
            
    #         # We evaluate both on the kfold and on the pre-set
    #         # We do the k-fold split for consistency, but we only use train and val
    #         train_df = full_df.iloc[train_index]
    #         kv_test_df = full_df.iloc[test_index]
    #         # Train on the cv, same as normal
    #         kv_cur_acc, kv_cur_f1 = self.mr_one_run_pre_split(train_df, kv_test_df, val_r)
    #         # Extra evaluation on the predefined test
    #         cur_acc, cur_f1 = self.mr_test(test_df)
            
    #         # Return all the results
    #         all_results.append((kv_cur_acc, kv_cur_f1, cur_acc, cur_f1))
            
    #     return(all_results)            
    
    
    #### Augmentation
    
    # Function for a dummy 10-fold cross validation with a data augmentation
    # dataframe
    # We need this specific function to ensure that:
    # - we do not use during training augmented examples based on the test
    # - we only use gold-standard examples for testing
    # 
    # Input - full df, aug df, ratio for splitting on val, number of runs
    def mr_kfold_aug(self, full_df, aug_df, val_r=0.25, num_runs=10, r_state = 42):
        
        # Initialize output
        all_results = []        
        
        # Run k-fold split
        kf = KFold(n_splits=num_runs, shuffle=True, random_state = r_state)
        
        # Run different splits
        for train_index, test_index in kf.split(full_df):
            train_df = full_df.iloc[train_index]
            test_df = full_df.iloc[test_index]
            
            
            # Get the augmented data
            # We only get the augments from the training set, discarding the
            # augments from the test set
            train_ids = train_df["Child_ID"].tolist()
            aug_data = aug_df[aug_df["Orig_ID"].isin(train_ids)].copy()
            
            # drop the orig_id column to avoid errors in merge
            aug_data.drop(["Orig_ID"],axis=1,inplace=True)
            
            # Merge and shuffle the train and aug data
            train_plus = shuffle(pd.concat([train_df,aug_data],axis=0, ignore_index=True))
            
            # Train and test with the new train
            cur_acc, cur_f1 = self.mr_one_run_pre_split(train_plus, test_df, val_r)
            
            all_results.append((cur_acc, cur_f1))
            
        return(all_results)        
        
    # Function for a dummy 10-fold cross validation with a data augmentation
    # dataframe and external test
    # Input - full df, aug df, test df ratio for splitting on val, number of runs
    def mr_kfold_aug_pre_split(self, full_df, aug_df, test_df, val_r=0.25, num_runs=10, r_state = 42):
        
        # Initialize output
        all_results = []        
        
        # Run k-fold split
        kf = KFold(n_splits=num_runs, shuffle=True, random_state = r_state)
        
        # Run different splits
        for train_index, test_index in kf.split(full_df):
            train_df = full_df.iloc[train_index]
            kv_test_df = full_df.iloc[test_index]
            
            
            # Get the augmented data
            # We only get the augments from the training set, discarding the
            # augments from the test set
            train_ids = train_df["Child_ID"].tolist()
            aug_data = aug_df[aug_df["Orig_ID"].isin(train_ids)].copy()
            
            # drop the orig_id column to avoid errors in merge
            aug_data.drop(["Orig_ID"],axis=1,inplace=True)
            
            # Merge and shuffle the train and aug data
            train_plus = shuffle(pd.concat([train_df,aug_data],axis=0, ignore_index=True))
            
            # Train and test with the new train
            kv_cur_acc, kv_cur_f1 = self.mr_one_run_pre_split(train_plus, kv_test_df, val_r)
            
            # Extra evaluation on the predefined test
            cur_acc, cur_f1 = self.mr_test(test_df)
            
            # Return all the results
            all_results.append((kv_cur_acc, kv_cur_f1, cur_acc, cur_f1))            
            
        return(all_results)              
    
            
            
            
