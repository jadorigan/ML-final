from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import VarianceThreshold

def selecao_feature(X, y, resp1):
    print('SHAPE ENTRADA')
    print (X.shape)
    
    if resp1 == 1:
        clf = ExtraTreesClassifier(n_estimators=100).fit(X, y)
        model = SelectFromModel(clf,prefit=True)
        X_new = model.transform(X)
    elif resp1 == 2:
        clf = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
        model = SelectFromModel(clf,prefit=True)
        X_new = model.transform(X)
    elif resp1 == 3:
        clf = VarianceThreshold(threshold=(.9 * (1 - .9)))
        X_new = clf.fit_transform(X)

    print('\nNEW SHAPE')
    print(X_new.shape)
    return X_new