import lib.utils as utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Feature Scaling
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error

from impressao import print_resultados
from selecaoCaracteristicas import selecao_feature

def Classifier_Gradient_Boosting(X, y, start_time):
    X_new = selecao_feature(X, y)
   
    # split data into train and test sets
    seed = 0
    test_size = 0.2
    n_estimators = 200

    # separação treino - teste: 80 - 20
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=test_size, random_state=seed)

    #shapes
    print('\nTraining Shape :',X_train.shape)
    print('Testing  Shape :',X_test.shape)

    # fit model no training data
    #Parâmetros: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    model = GradientBoostingClassifier(loss='exponential', learning_rate=0.1, random_state=0, n_estimators=n_estimators)
    model.fit(X_train, y_train)

    start_time = utils.get_time() # Tempo inicial

    mse = mean_squared_error(y_test, model.predict(X_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

    #test_score = np.zeros((n_estimators,), dtype=np.float64)

    #for i, y_pred in enumerate(model.staged_predict(X_test)):
    #    test_score[i] = model.loss_(y_test, y_pred)
        #print ('Test Score: '+ str(test_score[i]))

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    diff_time = utils.get_time_diff(start_time) # Tempo final

    print_resultados(model, y_test, predictions, diff_time, X_test)