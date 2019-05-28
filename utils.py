import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confussion_matrix(y_true, y_pred, classes, normalize=False, title=None,
                           cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confussion matrix'
        else:
            title = 'Confussion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    # classes = classes[unique_labels(y_true, y_pred).astype(np.int)]

    if normalize:
        cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots()
    image = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(image, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes, 
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    fmt = '.2f' if normalize else 'd'
    thresh =  cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center', color='white'
                    if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    return ax

