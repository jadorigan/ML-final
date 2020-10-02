import pandas as pd
import randomForest_mirtron_classifier as rf
import xgBoost_mirtron_classifier as xg
import gradientBoosting_mirtron_classifier as gb

dataset = pd.read_csv('data/dataset.csv')
start_time = 0
X = dataset.iloc[:, 0:94].values #atributos
y = dataset.iloc[:, 94].values #labels
    
if __name__ == "__main__":
   # rf.Random_Forest(X, y, start_time)
   # xg.Classifier_XGBoost(X, y, start_time)
    gb.Classifier_Gradient_Boosting(X, y, start_time)