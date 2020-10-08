from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_roc_curve
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from datetime import datetime

def get_time():
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H.%M.%S")
    return dt_string

def print_resultados(classifier, y_test, y_pred, diff_time, X_test, nome):
    
    print('\nClassifier: ')
    #print(classifier)
    #print('x: ', X_test)
    #print('y: ', y_test)

    print('\n:::RESULTS:::')
    
    print("\nMatriz de Confusão: \n", confusion_matrix(y_test,y_pred))
    
    print("\nClassification Report: \n", classification_report(y_test,y_pred))
    folder = "resultados/cls_report"
    output_file = folder + "/" + get_time() + " - " + nome + ".csv"
    pf = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
    pf.transpose
    pf.to_csv(output_file, index=True, header=True)
   
    print(f'\nTempo de Execução: {diff_time} s.')
    #print("\nAcurácia: ", accuracy_score(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAcurácia: : %.2f%%" % (accuracy * 100.0))

    mse = mean_squared_error(y_test, classifier.predict(X_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
    
    #print("\nImportância das Características: \n", classifier.feature_importances_)
    #######################################
    # Verificar se está certo Curva ROC #

    #plot_roc_curve(classifier, X_test, y_test) 
    #plt.show()
    ##########################################

    metrics.plot_roc_curve(classifier, X_test, y_test)  # doctest: +SKIP
    folder = "resultados/roc"
    output_file = folder + "/" + get_time() + " - " + nome + ".png"
    plt.savefig(output_file)
    #plt.show()

    metrics.plot_precision_recall_curve(classifier, X_test, y_test)
    folder = "resultados/precision_recall"
    output_file = folder + "/" + get_time() + " - " + nome + ".png"
    plt.savefig(output_file) 
    #plt.show()

    # Plot non-normalized confusion matrix
    titles_options = [("Without Normalization", None),("Normalized", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
        folder = "resultados/conf_mat"
        output_file = folder + "/" + get_time() + " - " + nome + "_" + title + ".png"
        plt.savefig(output_file)  
    #plt.show()
