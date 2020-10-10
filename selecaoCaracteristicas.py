from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import VarianceThreshold

def selecao_feature(X, y, resp1):
    print('\n********************************************************************')
    print('Shape Entrada: ', X.shape)
    if resp1 == 1:
        clf = ExtraTreesClassifier(n_estimators=100).fit(X, y)
        model = SelectFromModel(clf,prefit=True)
        X_new = model.transform(X)
        print('Extra Trees - New Shape: ', X_new.shape)
    elif resp1 == 2:
        clf = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
        model = SelectFromModel(clf,prefit=True)
        X_new = model.transform(X)
        print('LinearSVC - New Shape: ', X_new.shape)
    elif resp1 == 3:
        clf = VarianceThreshold(threshold=(.9 * (1 - .9)))
        X_new = clf.fit_transform(X)
        print('Variance Threshold - New Shape: ', X_new.shape)
    return X_new