from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

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