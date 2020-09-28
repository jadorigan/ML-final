import pandas as pd
import numpy as np
import lib.utils as utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Feature Scaling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_roc_curve
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/bill_authentication.csv')
start_time = 0

X = dataset.iloc[:, 0:4].values #atributos
y = dataset.iloc[:, 4].values #labels

# separação treino - teste: 80 - 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling (necessário?)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Treinamento e Predição
# 20 árvores (com 100 árvores tem o mesmo resultado)
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)
start_time = utils.get_time() # Tempo inicial
y_pred = classifier.predict(X_test)
diff_time = utils.get_time_diff(start_time) # Tempo final


print("Matriz de Confusão: \n", confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(f'Tempo de Execução: {diff_time} s.')
print("Acurácia: ", accuracy_score(y_test, y_pred))
print("Importância das Características: ", classifier.feature_importances_)
plot_roc_curve(classifier, X_test, y_test) #Curva ROC
#plt.show()