import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import math
from .utils import print_table
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def elbow_method(x_data, save=True, show=True):
    wcss_list = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++')
        kmeans.fit(x_data)
        wcss_list.append(kmeans.inertia_)
    figure = plt.figure()
    plt.plot(range(1, 11), wcss_list)
    plt.title('Graph of the Elbow method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('wcss_list')
    if save: plt.savefig(f'output/Elbow method.png')
    if show: plt.show()
    plt.close(figure)

def clustering(x_data, clusters, dimension, save=True, show=True):
    print('-'*19, "Clustering", '-'*19)
    kmeans = KMeans(n_clusters=clusters, init='k-means++')
    x_data = x_data[:, :dimension]
    y_predict = kmeans.fit_predict(x_data)

    print('Centroids: ')
    labels = ['count packets', 'full size', 'avg size', 'msd size',
              'full time', 'avg time', 'msd time', 'dir streams', 'rev streams']
    centroids = kmeans.cluster_centers_
    print_table(centroids, labels[:dimension])

    cols = len(x_data[0])
    max_combs = (cols * (cols - 1)) // 2
    plot_cols = 6
    plot_rows = math.ceil(max_combs / plot_cols)
    colors = ['#%02x%02x%02x' % (i, i, i)
              for i in range(0, 255, 255 // clusters)]
    plt.rcParams.update({'font.size': 10})
    figure, axs = plt.subplots(nrows=plot_rows, ncols=plot_cols, 
                            figsize=(18, 6), squeeze=False)
    figure.tight_layout()
    iter = 0
    # Отрисовка графиков зависимости labels[row] от labels[col]
    for row in range(cols):
        for col in range(row + 1, cols):
            _x = iter % plot_cols
            _y = iter // plot_cols
            axs[_y, _x].scatter(x_data[:, row], x_data[:, col], 
                                s=40, c=[colors[i] for i in y_predict])
            axs[_y, _x].scatter(centroids[:, row],
                                centroids[:, col], s=20, c='red')
            axs[_y, _x].set(xlabel=labels[row], ylabel=labels[col])
            axs[_y, _x].axes.xaxis.set_ticks([])
            axs[_y, _x].axes.yaxis.set_ticks([])
            iter += 1
    # Отключение пустых графиков в последней строчке
    if iter % plot_cols != 0:
        for _x in range(iter % plot_cols, plot_cols):
            axs[plot_rows - 1, _x].axis('off')
        
    if save: 
        file_name = f"Graph for clusters {clusters} and dimension {dimension}"
        plt.savefig('output/'+file_name+'.png')
    if show: plt.show()
    plt.close(figure)

def get_confusion_matrix(y_ideal, y_predict, labels, file_name='Confusion Matrix', save=True, show=True):
    print('-'*16, "Confusion matrix", '-'*16)
    conf_matrix = confusion_matrix(y_ideal, y_predict)
    print_table(conf_matrix, [], header=False, border=False)
    
    figure = plt.figure()
    sn.heatmap(conf_matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap="viridis")
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save: plt.savefig('output/'+file_name+'.png', dpi=400, bbox_inches="tight")
    if show: plt.show()
    plt.close(figure)

def get_classification_report(y_ideal, y_predict, labels):
    print('-'*13, "Classification  report", '-'*13)
    print("Count errors:", np.sum(y_ideal != y_predict))
    print("Accuracy: %.2f" % accuracy_score(y_ideal, y_predict))
    print("Other metrics:")
    p, r, f1, s = metrics.precision_recall_fscore_support(
        y_ideal, y_predict, labels=labels)
    _metrics = list(zip(labels, p, r, f1, s))
    print_table(_metrics, ["port", "precision", "recall", "f1-score", "count"])

def get_roc_curve(y_ideal, proba, file_name='ROC curve', show=True, save=True):
    fpr, tpr, _ = roc_curve(y_ideal, proba)
    figure = plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save: plt.savefig('output/'+file_name+'.png')
    if show: plt.show()
    plt.close(figure)

def classification(x_data, y_data, test_size=0.1, save=True, show=True):
    # Чтобы сделать адекватную ROC кривую (как и матрицу ошибок)
    # нужно чтобы в тестовой и обучающей выборке были как минимум по 1
    # одинаковому порту. Например можно избавиться от портов на которых
    # работал один поток, и разделить выборку как минимум по 1 в обучающую
    # и тестовую
    print('-'*17, "Classification", '-'*17)
    knn = KNeighborsClassifier()
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_size)
    y_predict = np.array(knn.fit(x_train, y_train).predict(x_test))
    print("Data size:", len(x_data))
    print("Train size:", len(x_train))
    print("Test size:", len(x_test))

    labels = unique_labels(y_test, y_predict)
    get_classification_report(y_test, y_predict, labels)
    get_confusion_matrix(y_test, y_predict, labels, save=save, show=show)

def prediction(data, save=True, show=True):
    print('-'*19, "Prediction", '-'*19)

    y, counts = np.unique(data, return_counts=True)
    pivot = y[counts == counts.max()][0]
    y_ideal = np.array(data == pivot, dtype=int)
    print("Pivot value:", pivot)

    len_data = len(data)
    print("Data size:", len_data)

    for window in np.linspace(1, int(len_data * 0.3), 3, dtype=int):
        for train_coeff in [0.7, 0.5, 0.4]:
            train_size = int(len_data * train_coeff)
            _y = [y_ideal[index - window:index] for index in range(window, train_size - 1)]
            log_reg = LogisticRegression()
            log_reg.fit(_y, np.asarray(y_ideal[window:train_size - 1], dtype=int).T)    

            predicts = []
            proba = []
            for index in range(train_size + 1, len_data):
                predict = log_reg.predict([y_ideal[index - window:index]])[0]
                predicts.append(1 if predict >= 0.5 else 0)
                proba.extend(log_reg.predict_proba([y_ideal[index - window:index]])[::, 1])

            file_name = ' for pivot {}, window size {} and train size {}'.format(pivot, window, train_size)
            print("Window size:", window, " Train size:", train_size, " Test size:", len_data - train_size)
            get_classification_report(y_ideal[train_size + 1:], predicts, labels=[0, 1])
            get_confusion_matrix(y_ideal[train_size + 1:], predicts, labels=[0, 1], 
                                file_name='Confusion Matrix ' + file_name, save=save, show=show)
            get_roc_curve(y_ideal[train_size + 1:], proba,
                          file_name='ROC curve ' + file_name, show=show, save=save)
