from src.utils import elapsed_time, print_table
from src.db_helper import *
from src.analysis_helper import *
import numpy as np
import argparse


def print_cluster_streams(cluster, streams):
    # Функция выводит кластер, кол-во потоков в кластере
    # и первые 10 потоков в этом кластере
    labels = ["ip source", "ip destination",
              "port source", "port destination"]
    print("Cluster:", cluster, " Count streams:", len(streams))
    print_table(streams, labels, rows_limit=10)


def analyse_clustering(streams, statistics, dimensions, max_clusters, labels, save=True, show=False):
    # Обходим все переданные размерности
    for dimension in dimensions:
        # Конечная таблица с количествами потоков в каждом кластере
        count_table = []
        print("Dimension:", dimension)
        # Применяем все переданные кол-ва кластеров
        for cluster in range(2, max_clusters + 1):
            print("Clusters:", cluster)
            # Получаем результат кластеризации данных
            predict = clustering(statistics, cluster,
                                 dimension, labels, save=save, show=show)
            # Формируем массив [[IP, порт отправителя/получаетя], кластер]
            cluster_streams = [[stream, y] for stream, y in zip(streams, predict)]
            # Сортируем массив по кластерам (от меньшего к большему)
            cluster_streams.sort(key=lambda y: y[1])
            # Добавляем последним элементом заглушку, с невозможным значением кластера
            cluster_streams.append([[], max_clusters])
            # Строка таблицы с количествами потоков в каждом кластере
            count_row = [cluster]
            # Значение предыдущего кластера
            prev_cluster = cluster_streams[0][1]
            # Потоки предыдущего кластера
            prev_streams = []
            for cluster_stream in cluster_streams:
                # Если значение кластера предыдущего потока != рассматриваемому
                if cluster_stream[1] != prev_cluster:
                    # Добавляем в строку кол-во потоков
                    count_row.append(len(prev_streams))
                    # Выводим таблицу с потоками предыдущего кластера
                    print_cluster_streams(prev_cluster, prev_streams)
                    # Обновляем переменную предыдущего кластера
                    prev_cluster = cluster_stream[1]
                    # Очищаем массив с потоками
                    prev_streams = []
                # Добавляем поток в массив потоков
                prev_streams.append(cluster_stream[0])
            # Дополняем - для вида матрицы
            count_row.extend('-' * (max_clusters + 1 - len(count_row)))
            # Добавляем строчку в таблицу
            count_table.append(count_row)
        print("Count of streams per cluster:")
        print_table(count_table, ["Count of clusters"] + list(range(max_clusters)))


def analyse_prediction(streams, min_timestamp, max_timestamp, count_intervals, ports, save=True, show=False):
    # Длина всего интервала
    length_interval = max_timestamp - min_timestamp
    # Длина каждого интервала
    delta = length_interval / count_intervals
    print("Length of all interval:", length_interval)
    print("Count of intervals:", count_intervals)
    print("Lenght of interval:", delta)

    # Подсчет уникальных портов
    y, counts = np.unique(ports, return_counts=True)
    # Выбор порта чаще всего использующегося
    pivot = y[counts == counts.max()][0]
    print("Port:", pivot)

    # Массив из 0 и 1, где 1 означает, что на этом
    # интервале начался поток с портом pivot
    intervals = [0] * count_intervals
    for stream in streams:
        # Если порт рассматрвиаемого порта не равен тому, что мы предсказываем
        # и если timestamp потока выходит за пределы, то пропускаем этот поток
        if stream[3] != pivot or stream[5] > max_timestamp:
            continue
        # Записываем 1 в интервал, что означает
        # что в этом интервале поток начал работать
        intervals[int((stream[5] - min_timestamp) / delta)] = 1
    print("Intervals:", intervals)
    prediction(intervals, save=save, show=show)


def action_db(args):
    elaps_time = 0.0
    if args.no_create is False:
        elaps_time += elapsed_time(pcap2db, args.pcap_name, args.db_name)
    if args.no_proc is False:
        elaps_time += elapsed_time(select_streams, args.db_name, args.limit)
        elaps_time += elapsed_time(append_statistics, args.db_name)
        elaps_time += elapsed_time(print_statistic, args.db_name)
    if args.no_create is False or args.no_proc is False:
        print(f"All elapsed time: {elaps_time:0.3f} seconds")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dn", "--db_name", default="database.db", help="Database filename")
    parser.add_argument("-pn", "--pcap_name", help="Pcap filename")
    parser.add_argument("-l", "--limit", help="Min count packets in thread", default=100, type=int)
    parser.add_argument("-cl", "--clusters", help="Max count of clusters", default=3, type=int)
    parser.add_argument("-dims", "--dimensions", nargs='*', help="Dimensions sample for clustering", default=[9])
    parser.add_argument("--no-create", help="Dont create new database file", action="store_true")
    parser.add_argument("--no-show", help="Dont show windows", action="store_true")
    parser.add_argument("--no-save", help="Dont save windows", action="store_true")
    parser.add_argument("--no-proc", help="Skip select streams and append statistic", action="store_true")
    args = parser.parse_args()

    if not args.no_create and args.pcap_name is None:
        print("No argument -pn/--pcap_name or --no-create")
        quit(1)

    try:
        dimensions = [int(dim) for dim in args.dimensions]
        args.dimensions = dimensions
    except ValueError:
        print("Invalid dimension")
        quit(1)

    return args


if __name__ == '__main__':
    args = parse_args()
    action_db(args)

    # Получение всех строк из таблицы streams
    streams = get_all_rows(args.db_name, "streams")
    # Массив с IP/портом отправителя/получателя
    info_streams = np.array(streams)[:, 1:5].astype(str)
    # Берем колонку с source port
    ports = np.array(info_streams)[:, 2].astype(int)
    # Отделяем статистику потоков
    statistics = np.array(streams)[:, 7:].astype(float)

    _save = not args.no_save
    _show = not args.no_show

    # Кластеризация
    labels = ['count packets', 'full size', 'avg size', 'msd size',
              'full time', 'avg time', 'msd time', 'dir streams', 'rev streams']
    analyse_clustering(info_streams, statistics, args.dimensions,
                       args.clusters, labels, save=_save, show=_show)
    # elbow_method(statistics, save=_save, show=_show)
    # clustering(statistics, args.clusters, args.dimension, labels, save=_save, show=_show)

    # Классификация
    classification(statistics, ports, save=_save, show=_show)

    # Прогнозирование
    packets = get_all_rows(args.db_name, "packets")
    packets_size = len(packets)
    min_timestamp = packets[0][5]
    max_timestamp = packets[packets_size - 1][5]

    analyse_prediction(streams, min_timestamp, max_timestamp, 1000, ports, save=_save, show=_show)

    # analyse_prediction(streams, min_timestamp, max_timestamp // 10, 100, ports, save=_save, show=_show)
