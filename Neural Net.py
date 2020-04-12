# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:12:33 2020

@author: Owner
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
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
    
    # Fix Data Types
    df.age = pd.to_numeric(df.age)
    df.TSH = pd.to_numeric(df.TSH)
    df.T3 = pd.to_numeric(df.T3)
    df.TT4 = pd.to_numeric(df.TT4)
    df.T4U = pd.to_numeric(df.T4U)
    df.FTI = pd.to_numeric(df.FTI)
    
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
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(features)
    features = ohe.transform(features).toarray()
    target = newDF.Class.values
    oe = OrdinalEncoder()
    oe.fit(target)
    target = oe.transform(target)
    target = to_categorical(target)
    return(features, target, newDF)

def create_model():
    # Build Neural Net
    model = keras.Sequential([
        keras.layers.Dense(100, activation='relu', input_dim = features.shape[1]),
        keras.layers.Dense(2, activation = 'softmax')
        ])
    # Specify Loss/Metric Functions
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[tf.keras.metrics.AUC()])
    return(model)

def train_and_evaluate_model(model, xTrain, yTrain, xTest, yTest):
    
    # Train Model
    modelFit = model.fit(xTrain, yTrain, epochs = 100)
    
    # Evaluate Model
    modelEval = modelFit.evaluate(xTest, yTest)
    
    return(modelEval)

if __name__ == "__main__":
    auc = []
    n_folds = 10
    features, target, newDF = load_data()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state = 24)

    for i, (train, test) in enumerate(skf.split(features, newDF['Class'])):
            print("Running Fold", i+1, "/", n_folds)
            model = None
            model = create_model()
            train_and_evaluate_model(model, features[train], target[train], features[test], target[test])
            auc.append(modelEval[1])
            print("AUC: ", modelEval[1])