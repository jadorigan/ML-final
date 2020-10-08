import lib.utils as utils
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from impressao import print_resultados
from selecaoCaracteristicas import selecao_feature

def Classifier_LDA(X, y, start_time):
    nome = "LDA"
    X_new = selecao_feature(X, y)
    
    # split data into train and test sets
    seed = 100
    test_size = 0.3

    # separação treino - teste: 80 - 20
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=test_size, random_state=seed)

    #shapes
    print('\nTraining Shape :',X_train.shape)
    print('Testing  Shape :',X_test.shape)

    # fit model no training data
    ## parametros do Modelo: https://xgboost.readthedocs.io/en/latest/python/python_api.html
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)

    #modelo
    print("\nMODELO")
    print(model)

    # make predictions for test data
    start_time = utils.get_time() # Tempo inicial
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    diff_time = utils.get_time_diff(start_time) # Tempo final
    
    print_resultados(model, y_test, predictions, diff_time, X_test, nome)