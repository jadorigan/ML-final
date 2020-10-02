from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_roc_curve
import matplotlib.pyplot as plt

def print_resultados(classifier, y_test, y_pred, diff_time, X_test):
    
    print('\n:::RESULTS:::')
    print("\nMatriz de Confusão: \n", confusion_matrix(y_test,y_pred))
    print("\nClassification Report: \n", classification_report(y_test,y_pred))
    print(f'\nTempo de Execução: {diff_time} s.')
    #print("\nAcurácia: ", accuracy_score(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAcurácia: : %.2f%%" % (accuracy * 100.0))
    print("\nImportância das Características: \n", classifier.feature_importances_)
    #######################################
    # Verificar se está certo Curva ROC #

    #plot_roc_curve(classifier, X_test, y_test) 
    #plt.show()
    ##########################################