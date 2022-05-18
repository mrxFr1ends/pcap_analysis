import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import math
from .utils import print_table
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
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
    figure = plt.figure()
    plt.plot(range(1, 11), wcss_list)
    plt.title('Graph of the Elbow method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('wcss_list')
    if save: plt.savefig(f'output/Elbow method.png')
    if show: plt.show()
    plt.close(figure)

def clustering(x_data, clusters, dimension, labels, save=True, show=True):
    print('-'*19, "Clustering", '-'*19)
    # Классификатор
    kmeans = KMeans(n_clusters=clusters)
    # Обрезаем данные до размерности dimension
    x_data = x_data[:, :dimension]
    # Обучаем классификатор и получаем распределения
    # потоков по кластерам
    y_predict = kmeans.fit_predict(x_data)

    # Выводим координаты центроидов в dimension пространстве
    print('Centroids: ')
    centroids = kmeans.cluster_centers_
    print_table(centroids, labels[:dimension])

    # Кол-во колонок
    cols = len(x_data[0])
    # Кол-во комбинаций параметров между собой
    max_combs = (cols * (cols - 1)) // 2
    # Кол-во графиков по оси X
    plot_cols = 6
    # Кол-во графиков по оси Y
    plot_rows = math.ceil(max_combs / plot_cols)
    # Массив цветов от черного до белого
    colors = ['#%02x%02x%02x' % (i, i, i)
              for i in range(0, 255, 255 // clusters)]
    # Размер текста
    plt.rcParams.update({'font.size': 10})
    figure, axs = plt.subplots(nrows=plot_rows, ncols=plot_cols, 
                            figsize=(18, 6), squeeze=False)
    iter = 0
    # Отрисовка графиков зависимости labels[row] от labels[col]
    for row in range(cols):
        for col in range(row + 1, cols):
            _x = iter % plot_cols
            _y = iter // plot_cols
            # Отрисовка потоков
            axs[_y, _x].scatter(x_data[:, row], x_data[:, col], 
                                s=40, c=[colors[i] for i in y_predict])
            # Отрисовка центроидов
            axs[_y, _x].scatter(centroids[:, row],
                                centroids[:, col], s=20, c='red')
            # Установка названия осей
            axs[_y, _x].set(xlabel=labels[row], ylabel=labels[col])
            axs[_y, _x].axes.xaxis.set_ticks([])
            axs[_y, _x].axes.yaxis.set_ticks([])
            iter += 1
    # Отключение пустых графиков в последней строчке
    if iter % plot_cols != 0:
        for _x in range(iter % plot_cols, plot_cols):
            axs[plot_rows - 1, _x].axis('off')
        
    title = f"Graph for clusters {clusters} and dimension {dimension}"
    figure.suptitle(title)
    figure.tight_layout()
    # Сохранение графиков
    if save: plt.savefig('output/'+title+'.png')
    # Отображение графиков
    if show: plt.show()
    plt.close(figure)
    # Возвращение результата
    return y_predict

# Функция получения матрицы ошибок
def get_confusion_matrix(y_ideal, y_predict, labels, file_name='Confusion Matrix', save=True, show=True):
    print('-'*16, "Confusion matrix", '-'*16)
    # Получаем матрицу ошибок
    conf_matrix = confusion_matrix(y_ideal, y_predict, labels=labels)
    # Выводим матрицу ошибок в консоль
    print_table(conf_matrix, [], header=False, border=False)
    # Отрисовываем матрицу ошибок
    figure = plt.figure()
    hm = sn.heatmap(conf_matrix, annot=True, cmap="viridis", fmt="g")
    hm.set_xticklabels(labels, rotation=90, horizontalalignment='right')
    hm.set_yticklabels(labels, rotation=0, horizontalalignment='right')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(file_name, fontsize=18)
    figure.tight_layout()
    # Если save = True сохраняем матрицу ошибок в файл png
    if save: plt.savefig('output/'+file_name+'.png', dpi=400, bbox_inches="tight")
    # Если show = True отображаем отрисованную матрицу
    if show: plt.show()
    plt.close(figure)

# Функция вывода основных метрик в консоль
def get_classification_report(y_ideal, y_predict, labels):
    print('-'*13, "Classification  report", '-'*13)
    print("Count errors:", sum(1 for x, y in zip(y_ideal, y_predict) if x == y))
    print("Accuracy: %.2f" % accuracy_score(y_ideal, y_predict))
    print("Other metrics:")
    # Получаем метрики precision, recall, f1-score, count
    p, r, f1, c = metrics.precision_recall_fscore_support(
        y_ideal, y_predict, labels=labels)
    # Объединяем метрики с их портами
    _metrics = list(zip(labels, p, r, f1, c))
    # Выводим таблицу с метриками
    print_table(_metrics, ["port", "precision", "recall", "f1-score", "count"])

def get_roc_curves(y, proba, classes, file_name='ROC curve', show=True, save=True):
    figure = plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(file_name, fontsize=18)

    for iter, _class in enumerate(classes):
        # Массив 0 и 1, где 1 это y == _class
        _y_test = [0 if _y != _class else 1 for _y in y]
        # Получение fpr и tpr
        fpr, tpr, _ = roc_curve(_y_test, proba[:, iter])
        # Рассчитываем площадь под графиком
        roc_auc = auc(fpr, tpr)
        # Если площадь посчиталась неверно, пропускаем
        if math.isnan(roc_auc): continue
        # Рисуем график ROC-кривой
        plt.plot(fpr, tpr, label=f"{_class} (area = {roc_auc:.2f})")

    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.legend()
    if save: plt.savefig('output/'+file_name+'.png')
    if show: plt.show()
    plt.close(figure)

def classification(x_data, y_data, test_size=0.1, save=True, show=True):
    print('-'*17, "Classification", '-'*17)
    # Создает классификатор
    knn = KNeighborsClassifier()
    # Разделяем всю выборку на обучающую и тестовую
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_size)    
    print("Data size:", len(x_data))
    print("Train size:", len(x_train))
    print("Test size:", len(x_test))

    # Обучаем классификатор
    knn.fit(x_train, y_train)
    # Получаем предсказания
    y_predict = knn.predict(x_test)
    # Получаем вероятности предсказаний
    y_proba = knn.predict_proba(x_test)
    # Получение ROC-кривых
    get_roc_curves(y_predict, y_proba, knn.classes_, 
                   file_name='Classification ROC curve',
                   save=save, show=show)
    # Получаем массив уникальных портов из тестовой и предсказанной выборки
    labels = unique_labels(y_test, y_predict)
    # Выводим в консоль основные метрики
    get_classification_report(y_test, y_predict, labels)
    # Вызываем функцию получения матрицы ошибок
    get_confusion_matrix(y_test, y_predict, labels, save=save, show=show)

def get_roc_curve(y_ideal, proba, file_name='ROC curve', show=True, save=True):
    fpr, tpr, _ = roc_curve(y_ideal, proba)
    figure = plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(file_name, fontsize=18)
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig('output/'+file_name+'.png')
    if show:
        plt.show()
    plt.close(figure)

def prediction(data, save=True, show=True):
    print('-'*19, "Prediction", '-'*19)
    len_data = len(data)
    print("Data size:", len_data)

    
    # Выборка разных коэффициентов размеров обучающей выборки
    for train_coeff in [0.7, 0.1, 0.05]:
        # Размер обучающей выборки
        train_size = int(len_data * train_coeff)
        # Выборка разных размеров окон (от 1 до 30 процентов от размера всех данных)
        for window in np.linspace(1, int(train_size * 0.7), 3, dtype=int):
            # Матрица обучающей выборки
            _y = [data[index - window:index] for index in range(window, train_size - 1)]
            # Классификатор
            mlpc = MLPClassifier(shuffle=False)
            # Обучаем классификатор
            mlpc.fit(_y, np.asarray(data[window:train_size - 1], dtype=int).T)    
            # Массив предсказаний
            predicts = []
            # Массив вероятностей предсказаний
            proba = []
            # Проходим по тестовой выборке
            for index in range(train_size, len_data):
                # Производим предсказание, добавляя его в массив
                predicts.append(mlpc.predict([data[index - window:index]])[0])
                # Добавляем вероятность предсказаний в массив
                proba.extend(mlpc.predict_proba([data[index - window:index]])[::, 1])

            file_name = ' for window size {} and train size {}'.format(window, train_size)
            print("Window size:", window, " Train size:", train_size, " Test size:", len_data - train_size)
            # Получение основных метрик
            get_classification_report(data[train_size:], predicts, labels=[0, 1])
            # Получение матрицы ошибок
            get_confusion_matrix(data[train_size:], predicts, labels=[0, 1], 
                                file_name='Confusion Matrix ' + file_name, save=save, show=show)
            # Получение ROC-кривой
            get_roc_curve(data[train_size:], proba,
                            file_name='ROC curve ' + file_name, show=show, save=save)
