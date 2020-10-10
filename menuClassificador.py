import randomForest_mirtron_classifier as rf
import xgBoost_mirtron_classifier as xg
import gradientBoosting_mirtron_classifier as gb
import svm_mirtron_classifier as sv
import lda_mirtron_classifier as lda
import bayes_mirtron_classifier as bayes
import log_reg_mirtron_classifier as logreg
import knn_mirtron_classifier as knn

def menuSelClass(resp1, X, y):
    while True:
        print('================ Seleção de Classificadores ==================')
        print('1 - Random Forest')
        print('2 - XGBoost')
        print('3 - Gradient Boosting')
        print('4 - SVM')
        print('5 - LDA')
        print('6 - Bayes')
        print('7 - Logistic Regression')
        print('8 - KNN')
        print('0 - Sair')
        resp2 = float(input("Escolha: "))
        if resp2==0:
            break
        elif resp2==1:
            rf.Random_Forest(X, y, resp1)
        elif resp2 ==2:
            xg.Classifier_XGBoost(X, y, resp1)
        elif resp2 ==3:
            gb.Classifier_Gradient_Boosting(X, y, resp1)
        elif resp2 ==4:
            sv.SVM(X, y, resp1)
        elif resp2 ==5:
            lda.Classifier_LDA(X, y, resp1)
        elif resp2 ==6:
            bayes.Classifier_Bayes(X, y, resp1)
        elif resp2 ==7:
            logreg.Classifier_LogisticRegression(X, y, resp1)
        elif resp2 ==8:
            knn.Classifier_KNN(X, y, resp1)
        break
    return resp2