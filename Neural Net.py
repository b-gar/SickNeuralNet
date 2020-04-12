# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:12:33 2020

@author: Owner
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras

def load_data():
    # Read in Data
    df = pd.read_csv('C:/Users/Owner/OneDrive - UWM/Spring 2020/COMPSCI 657/sick.csv')
    
    # Drop Column with all nan and Rows with Mainly nan
    df = df.drop('TBG', axis=1)
    df = df.dropna(thresh=(len(df.columns)/2))
    
    # Clean Messy Values
    df = df.replace('?', np.NaN)
    df = df.replace('f', False)
    df = df.replace('t', True)
    df = df.replace('negative', False)
    df = df.replace('sick', True)
    df = df[pd.notnull(df['sex'])]
    
    # OneHotEncoder Instance
    enc = OneHotEncoder(handle_unknown='ignore')
    
    # One Hot Encode for Category Columns
    sexCode = pd.DataFrame(enc.fit_transform(df[['sex']]).toarray())
    refCode = pd.DataFrame(enc.fit_transform(df[['referral source']]).toarray())
    
    # Change Column Names, Join DF's, and Drop Old Columns
    sexCode.columns = ['Female', 'Male']
    refCode.columns = ['STMW', 'SVHC', 'SVHD', 'SVI', 'other']
    df = pd.concat([df, sexCode, refCode], axis = 1)
    df = df.drop('sex', 1)
    df = df.drop('referral source', 1)
    
    # Check dtypes
    df.dtypes
    
    # Fix Data Types
    df.age = pd.to_numeric(df.age)
    df.TSH = pd.to_numeric(df.TSH)
    df.T3 = pd.to_numeric(df.T3)
    df.TT4 = pd.to_numeric(df.TT4)
    df.T4U = pd.to_numeric(df.T4U)
    df.FTI = pd.to_numeric(df.FTI)
    
    # Fix Data Types for Newer Columns
    df.STMW = df.STMW.replace(0, False).replace(1, True)
    df.SVHC = df.SVHC.replace(0, False).replace(1, True)
    df.SVHD = df.SVHD.replace(0, False).replace(1, True)
    df.SVI = df.SVI.replace(0, False).replace(1, True)
    df.other = df.other.replace(0, False).replace(1, True)
    df.Female = df.Female.replace(0, False).replace(1, True)
    df.Male = df.Male.replace(0, False).replace(1, True)
    
    
    # Imputation for Missing Values
    numericals = ['age','TSH', 'T3', 'TT4', 'T4U', 'FTI']
    catImpDF = df.drop(numericals, axis = 1)
    numImpDF = df[numericals]
    
    catImp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    numImp = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    catImpDFTransformed = pd.DataFrame(catImp.fit_transform(catImpDF))
    catImpDFTransformed.columns = catImpDF.columns
    numImpDFTransformed = pd.DataFrame(numImp.fit_transform(numImpDF))
    numImpDFTransformed.columns = numImpDF.columns
    
    newDF = pd.concat([catImpDFTransformed, numImpDFTransformed], axis = 1)
    
    # Get Features and Target into Array for Neural Net
    features = newDF.drop('Class', axis = 1).values
    target = newDF.Class.values
    return(features, target, newDF)

def create_model():
    # Build Neural Net
    model = keras.Sequential([
        keras.layers.Dense(100, activation='relu', input_shape = features.shape[1]),
        keras.layers.Dense(2, activation = 'softmax')
        ])
    # Specify Loss/Metric Functions
    model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=[tf.keras.metrics.AUC()])
    return(model)

def train_and_evaluate__model(model, xTrain, yTrain, xTest, yTest):
    
    # Train Model
    modelFit = model.fit(xTrain, yTrain, epochs = 100)
    
    # Evaluate Model
    modelEval = model.evaluate(xTest, yTest)

if __name__ == "__main__":
    auc = []
    n_folds = 10
    features, target, newDF = load_data()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state = 24)

    for i, (train, test) in enumerate(skf.split(features, newDF['Class'])):
            print("Running Fold", i+1, "/", n_folds)
            model = None # Clearing the NN.
            model = create_model()
            train_and_evaluate_model(model, features[train], target[train], features[test], target[test])
            auc.append(modelEval[1])
            print("AUC: ", modelEval[1])