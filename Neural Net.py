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
    return(df)

def clean_data(df):
    # Drop Column with all nan and Rows with Mainly nan
    df = df.drop('TBG', axis=1)
    df = df.dropna(thresh=(len(df.columns)/2))
    
    # Clean Messy Values
    df = df.replace('?', np.NaN)
    df = df.replace('f', 0)
    df = df.replace('t', 1)
    df = df.replace('negative', 0)
    df = df.replace('sick', 1)
    
    # Fix Data Types for Imputation
    df.age = pd.to_numeric(df.age)
    df.TSH = pd.to_numeric(df.TSH)
    df.T3 = pd.to_numeric(df.T3)
    df.TT4 = pd.to_numeric(df.TT4)
    df.T4U = pd.to_numeric(df.T4U)
    df.FTI = pd.to_numeric(df.FTI)
    return(df)

def impute_data(df):    
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
    
    dfnew = pd.concat([catImpDFTransformed, numImpDFTransformed], axis = 1)
    return(dfnew)

def onehotencode_data(df):
    # OneHotEncoder Instance
    enc = OneHotEncoder(handle_unknown='ignore')
    
    # One Hot Encode for Category Columns
    sexCode = pd.DataFrame(enc.fit_transform(df[['sex']]).toarray())
    refCode = pd.DataFrame(enc.fit_transform(df[['referral source']]).toarray())

    # Change Column Names, Join DF's, and Drop Old Columns
    sexCode.columns = ['Female', 'Male']
    refCode.columns = ['STMW', 'SVHC', 'SVHD', 'SVI', 'other']
    newDF = pd.concat([df, sexCode, refCode], axis = 1)
    newDF = newDF.drop('sex', 1)
    newDF = newDF.drop('referral source', 1)
    
    newDF = newDF.iloc[:,:21].astype(int)
    return(newDF)

def getxy(df):
    # Get Features and Target into Array for Neural Net
    features = df.drop('Class', axis = 1).values
    target = df.Class.values
    return(features, target, df)

def create_model(trainArray):
    # Build Neural Net
    model = keras.Sequential([
        keras.layers.Dense(100, activation='relu', input_dim = trainArray.shape[1]),
        keras.layers.Dense(1, activation = 'hard_sigmoid')
        ])
    # Specify Loss/Metric Functions
    model.compile(optimizer='rmsprop', loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()])
    return(model)

def train_eval_model(model, xTrain, yTrain, xTest, yTest):
    
    # Train Model
    modelFit = model.fit(xTrain, yTrain, epochs = 100)
    
    # Evaluate Model
    modelEval = model.evaluate(xTest, yTest)
    return(modelEval)

# Main Block
if __name__ == "__main__":

    data1 = load_data()
    data2 = clean_data(data1)
    data3 = impute_data(data2)
    data4 = onehotencode_data(data3)
    features, target, finalDF = getxy(data4)
    
    nfolds = 10
    skf = StratifiedKFold(n_splits = nfolds, shuffle = True, random_state = 24)
    auc = []
    
    for i, (train, test) in enumerate(skf.split(features, target)):
        print('Fold ', i+1, '/', nfolds)
        model = None
        model = create_model(features[train])
        fitModel = train_eval_model(model, features[train], target[train], features[test], target[test])
        auc.append(fitModel[1])
        print("Average AUC: ", fitModel[1])