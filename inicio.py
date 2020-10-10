import pandas as pd
import os
from menuCaracteristicas import menuSelCarac
from menuClassificador import menuSelClass

#dataset = pd.read_csv('data/dataset.csv')
#dataset = pd.read_csv('data/dataset_mat.csv')
dataset = pd.read_csv('data/dataset_prec.csv')

start_time = 0
X = dataset.iloc[:, 0:94].values #atributos
y = dataset.iloc[:, 94].values #labels
    
if __name__ == "__main__":
   os.system('cls')
   while True:
      resp1 = menuSelCarac()
      if resp1 == 0:
         break
      resp2 = menuSelClass(resp1, X, y, start_time)
      if resp2 == 0:
         break