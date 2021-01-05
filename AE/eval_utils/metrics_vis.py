import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

from eval_utils.tensorboard_utils import roc_figure, pr_curve_figure


def scores_histogram(labels, anomaly_scores):
    # plot histogram w.r.t label
    normal = anomaly_scores[labels == 0]
    defect = anomaly_scores[labels == 1]
    plt.hist(normal, bins=100, alpha=0.5, label='normal', density=True)
    plt.hist(defect, bins=100, alpha=0.5, label='defect', density=True)
    plt.legend(loc='upper right')
    plt.savefig('./logs/scores_histogram.png')
    plt.show()
    plt.clf()


def plot_classification_report(scores, labels):
    # plot histograms of anoamly scores w.r.t to label
    scores_histogram(labels, scores)

    # plot precision recall curve
    pr_curve_figure(labels, scores, show=True)

    # plot roc curve
    roc_figure(labels, scores, show=True)


def display_confusion_matrix(predictions, labels):
    cm = confusion_matrix(labels, predictions)
    print(classification_report(labels, predictions, target_names=['Normal', 'Anomaly']))
    ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Anomaly']).plot(values_format='d')
    plt.savefig('./logs/confusion_matrix.png')
    plt.show()
    plt.clf()
