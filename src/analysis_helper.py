import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def elbow_method(x_data, save=True, show=True):
    wcss_list = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++')
        kmeans.fit(x_data)
        wcss_list.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss_list)
    plt.title('Graph of the Elobw method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('wcss_list')
    if save: plt.savefig('output/elbow_method.png')
    if show: plt.show()

def clustering(x_data, count_clusters, save=True, show=True):
    print('-'*19, "Clustering", '-'*19)
    kmeans = KMeans(n_clusters=count_clusters, init='k-means++')
    y_predict = kmeans.fit_predict(x_data)
    centroids = kmeans.cluster_centers_
    np.set_printoptions(precision=3, suppress=True)
    print('Centroids: ')
    [print(i) for i in centroids]

    cols = len(x_data[0])
    titles = ['count packets', 'full size', 'avg size', 'msd size',
              'full time', 'avg time', 'msd time', 'dir streams', 'rev streams']
    colors = ['#%02x%02x%02x' % (i, i, i)
              for i in range(0, 255, 255 // count_clusters)]

    plt.rcParams.update({'font.size': 10})
    fig, axs = plt.subplots(6, 6, figsize=(18, 6))
    fig.tight_layout()
    iter = 0
    # Отрисовка графиков зависимости titles[row] от titles[col]
    for row in range(cols):
        for col in range(row + 1, cols):
            _x = iter % 6
            _y = iter // 6
            axs[_y, _x].scatter(x_data[:, row], x_data[:, col], s=40,
                                c=[colors[i] for i in y_predict])
            axs[_y, _x].set(xlabel=titles[row], ylabel=titles[col])
            axs[_y, _x].axes.xaxis.set_ticks([])
            axs[_y, _x].axes.yaxis.set_ticks([])
            axs[_y, _x].scatter(centroids[:, row],
                                centroids[:, col], s=20, c='red')
            iter += 1
    if save: plt.savefig('output/clusters.png')
    if show: plt.show()

def get_confusion_matrix(y_ideal, y_predict, labels, title='Confusion Matrix', save=True, show=True):
    print('-'*16, "Confusion matrix", '-'*16)
    conf_matrix = confusion_matrix(y_ideal, y_predict)
    _, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.viridis)
    table = PrettyTable()
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, conf_matrix[i, j], va='center',
                    ha='center', size='xx-large', c='w')
        table.add_row(conf_matrix[i])
    print(table.get_string(header=False, border=False))
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(title, fontsize=18)
    if save: plt.savefig('output/'+title+'.png')
    if show: plt.show()

def get_classification_report(y_ideal, y_predict, labels):
    print('-'*13, "Classification  report", '-'*13)
    print("Count errors:", np.sum(y_ideal != y_predict))
    print("Accuracy: %.2f" % accuracy_score(y_ideal, y_predict))
    print("Other metrics:")
    p, r, f1, s = metrics.precision_recall_fscore_support(
        y_ideal, y_predict, labels=labels)
    _metrics = zip(labels, p, r, f1, s)
    table = PrettyTable(["port", "precision", "recall", "f1-score", "count"])
    table.float_format = '.2'
    for row in _metrics:
        table.add_row(row)
    print(table)

def get_roc_curve(y_ideal, proba, title='ROC curve', show=True, save=True):
    fpr, tpr, _ = roc_curve(y_ideal, proba)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    if save: plt.savefig('output/'+title+'.png')
    if show: plt.show()

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

def prediction(x_data, y_data, save=True, show=True):
    print('-'*19, "Prediction", '-'*19)
    y, counts = np.unique(y_data, return_counts=True)
    pivot = y[counts == counts.max()][0]
    print("Pivot value:", pivot)
    y_ideal = np.array(y_data == pivot, dtype=int)
    for window in np.linspace(1, int(len(x_data) * 0.9), 5, dtype=int):
        predicts = []
        proba = []
        classificator = MLPClassifier(max_iter=1)
        for index in range(window, len(x_data)):
            _x = x_data[index - window:index]
            _y = y_ideal[index - window:index]
            x_predict = x_data[index]
            predict = classificator.fit(_x, _y).predict([x_predict])
            predicts.extend(predict)
            proba.extend(classificator.predict_proba([x_predict])[::,1])
        
        title = ' for pivot {} and window size {}'.format(pivot, window)
        print("Window size:", window)
        get_classification_report(y_ideal[window:], predicts, labels=[0, 1])
        get_confusion_matrix(y_ideal[window:], predicts, labels=[0, 1], 
                             title='Confusion matrix' + title, save=save, show=show)
        # TODO: Зачем это в prediction? Пока не понял. Нужно сделать в classification
        get_roc_curve(y_ideal[window:], proba, title='ROC curve' + title, show=show, save=save)
