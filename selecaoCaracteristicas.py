from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

def selecao_feature(X, y):
    print('SHAPE ENTRADA')
    print (X.shape)
    
    #clf = ExtraTreesClassifier(n_estimators=100).fit(X, y)
    clf = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)

    #print("\n FEATURE IMPORTANCE")
    #print(clf.feature_importances_)
    
    model = SelectFromModel(clf,prefit=True)
    X_new = model.transform(X)

    print('\nNEW SHAPE')
    print(X_new.shape)
    return X_new