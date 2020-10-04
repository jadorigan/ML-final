import pandas as pd
import randomForest_mirtron_classifier as rf
import xgBoost_mirtron_classifier as xg
import gradientBoosting_mirtron_classifier as gb
import os

dataset = pd.read_csv('data/dataset.csv')
#dataset = pd.read_csv('data/dataset_mat.csv')
#dataset = pd.read_csv('data/dataset_prec.csv')
start_time = 0
X = dataset.iloc[:, 0:94].values #atributos
y = dataset.iloc[:, 94].values #labels
    
if __name__ == "__main__":
   os.system('cls')
   while True:
      print('\n ================ Seleção de Classificadores ================== \n')
      print('1 - Random Forest')
      print('2 - XGBoost')
      print('3 - Gradient Boosting')
      print('0 - Sair')
      q = float(input("Escolha: "))
      if q==0:
         break
      elif q==1:
         rf.Random_Forest(X, y, start_time) # 71,5%
      elif q ==2:
         xg.Classifier_XGBoost(X, y, start_time) # 77,1%
      elif q ==3:
         gb.Classifier_Gradient_Boosting(X, y, start_time) # 78,2%