from src.utils import elapsed_time
from src.db_helper import *
from src.analysis_helper import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dn", "--db_name", default="database.db", help="Database filename")
    parser.add_argument("-pn", "--pcap_name", help="Pcap filename")
    parser.add_argument("-l", "--limit", help="Min count packets in thread", default=100, type=int)
    parser.add_argument("-cl", "--clusters", help="Count of clusters", default=3, type=int)
    parser.add_argument("-dim", "--dimension", help="Dimension sample for clustering", default=9, type=int)
    parser.add_argument("--no-create", help="Dont create new database file", action="store_true")
    parser.add_argument("--no-show", help="Dont show windows", action="store_true")
    parser.add_argument("--no-save", help="Dont save windows", action="store_true")
    parser.add_argument("--no-proc", help="Skip select streams and append statistic", action="store_true")
    args = parser.parse_args()

    if not args.no_create and args.pcap_name is None:
        print("No argument -pn/--pcap_name or --no-create")
        quit(1)
    _save = not args.no_save
    _show = not args.no_show

    elaps_time = 0.0
    if args.no_create == False:
        elaps_time += elapsed_time(pcap2db, args.pcap_name, args.db_name)
    if args.no_proc == False:
        elaps_time += elapsed_time(select_streams, args.db_name, args.limit)
        elaps_time += elapsed_time(append_statistics, args.db_name)
        elaps_time += elapsed_time(print_statistic, args.db_name)
    if args.no_create == True and args.no_proc == True:
        print(f"All elapsed time: {elaps_time:0.3f} seconds")
    
    # Получение всех строк из таблицы streams
    # streams = get_all_rows(args.db_name, "streams")
    streams = get_all_rows("database.db", "streams")
    # Убираем первые 5 колонок
    x = np.array(streams)[:, 5:].astype(float)

    # Кластеризация
    elbow_method(x, save=_save, show=_show)
    clustering(x, args.clusters, args.dimension, save=_save, show=_show)
    
    # Классификация
    # Берем колонку с source port
    ports = np.array(streams)[:, 3].astype(int)
    classification(x, ports, save=_save, show=_show)

    # Прогнозирование
    data = get_ports_sorted_by_timestamp(args.db_name)
    prediction(data, save=_save, show=_show)
