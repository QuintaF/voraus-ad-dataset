
from typing import List
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def compute_metrics(dfn: DataFrame, fpr: List[float], tpr: List[float], auroc: float, dataset: str, category, thresholds: List[float], pos_label, plot: bool) -> tuple[float, float, float, float]:
    '''
    computes various metrics to evaluate the algorithm performance
    and plots the results.

    :params dfn: data scores and label
    :params fpr: false positive rate
    :params tpr: true positive rate
    :params auroc: auroc score 
    :params dataset: name of the dataset evaluated
    :params category: which attack samples are being evaluated
    :params thresholds: thresholds used to compute fpr and tpr
    :params pos_label: label used to recognize attacks
    :params plot: show or hide plots

    :returns: (precision, recall, accuracy, f_score)
    '''
    if plot:
        plt.figure()
        plt.title(f'Receiver Operating Characteristic Attack {category}')
        plt.grid(visible=True)
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auroc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

    # best threshold
    v = np.absolute(fpr - tpr)
    best = thresholds[np.argmax(v)]

    # confusion matrix                      
    gold = ((dfn["anomaly"].to_numpy()) == pos_label).astype(int)
    predicted = ((dfn["score"].to_numpy()) > best).astype(int)
    conf_matrix = metrics.confusion_matrix(gold, predicted)

    if plot:
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ["Normal", "Attack"])
        cm_display.plot()
        plt.title(f'{dataset}, Attack: {category}')
        plt.ylabel("True Category")
        plt.xlabel("Predicted Category")
        plt.show()

    # scores
    accuracy = metrics.accuracy_score(gold, predicted, normalize= True)
    precisions, recalls, f_scores, _ = metrics.precision_recall_fscore_support(gold, predicted)
    print(f"Metrics for {dataset}, Attack: {category}")
    print(f"\tPrecision: {precisions[0]:.2f}, Recall: {recalls[0]:.2f}, Accuracy: {accuracy:.2f}, F1-Score: {f_scores[0]:.2f}")
            
    return precisions[0], recalls[0], accuracy, f_scores[0]
