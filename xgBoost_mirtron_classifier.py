import lib.utils as utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Feature Scaling
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

from impressao import print_resultados

#----------------------------------------------------------
# SELEÇÃO DE FEATURE
#----------------------------------------------------------
def selecao_feature(X, y):
    print('SHAPE ENTRADA')
    print (X.shape)
    clf = ExtraTreesClassifier(n_estimators=100)
    clf = clf.fit(X,y)
    print("\n FEATURE IMPORTANCE")
    print(clf.feature_importances_)
    model = SelectFromModel(clf,prefit=True)
    X_new = model.transform(X)
    print('\nNEW SHAPE')
    print(X_new.shape)
    return X_new

#----------------------------------------------------------
# PREDIÇÃO - XGBoost
#----------------------------------------------------------
def Classifier_XGBoost(X, y, start_time):
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
    model = XGBClassifier(random_state=1,learning_rate=0.01,max_depth=6, objective ='reg:logistic')
    model.fit(X_train, y_train)

    #modelo
    print("\nMODELO")
    print(model)

    # make predictions for test data
    start_time = utils.get_time() # Tempo inicial
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    diff_time = utils.get_time_diff(start_time) # Tempo final
    
    print_resultados(model, y_test, predictions, diff_time, X_test)