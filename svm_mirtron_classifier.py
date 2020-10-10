import lib.utils as utils
from sklearn.model_selection import train_test_split
from sklearn import svm #svm
from impressao import print_resultados
from selecaoCaracteristicas import selecao_feature

def SVM(X, y, resp1):
    nome = "SVM"
    X_new, nomeFeature  = selecao_feature(X, y, resp1)
    
    # separação treino - teste: 80 - 20
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=0)

    # Treinamento e Predição
    classifier = svm.SVC(kernel='rbf', C=5,gamma = 0.0001,probability=True)
    classifier.fit(X_train, y_train)
    start_time = utils.get_time() # Tempo inicial
    y_pred = classifier.predict(X_test)
    diff_time = utils.get_time_diff(start_time) # Tempo final
    
    print_resultados(classifier, y_test, y_pred, diff_time, X_test, nome, nomeFeature)