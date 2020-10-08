import pandas as pd
import randomForest_mirtron_classifier as rf
import xgBoost_mirtron_classifier as xg
import gradientBoosting_mirtron_classifier as gb
import svm_mirtron_classifier as sv
import lda_mirtron_classifier as lda
import bayes_mirtron_classifier as bayes
import log_reg_mirtron_classifier as logreg
import knn_mirtron_classifier as knn
import os

#dataset = pd.read_csv('data/dataset.csv')
#dataset = pd.read_csv('data/dataset_mat.csv')
dataset = pd.read_csv('data/dataset_prec.csv')

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
      print('4 - SVM')
      print('5 - LDA')
      print('6 - Bayes')
      print('7 - Logistic Regression')
      print('8 - KNN')
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
      elif q ==4:
         sv.SVM(X, y, start_time) # 72,6%
      elif q ==5:
         lda.Classifier_LDA(X, y, start_time) # 77,8%
      elif q ==6:
         bayes.Classifier_Bayes(X, y, start_time) # 76,3%
      elif q ==7:
         logreg.Classifier_LogisticRegression(X, y, start_time) # 78,6%
      elif q ==8:
         knn.Classifier_KNN(X, y, start_time) # 76,2%