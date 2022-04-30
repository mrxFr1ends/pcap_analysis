from prettytable import PrettyTable
import dpkt
from sqlite3 import *
from .utils import printProgressBar
import socket
from math import sqrt

# Функция парсинга pcap файла, возвращает массив прочитанных пакетов
def read_pcap(file_name):
    print("Read pcap file ...")
    with open(file_name, 'rb') as file:
        return dpkt.pcap.Reader(file).readpkts()

# Функция создания таблицы с прочитанными пакетами
def pcap2db(file_name, db_name="database.db"):
    # Очистка базы данных
    with open(db_name, 'w'): pass
    con = connect(db_name)
    cur = con.cursor()

    print("Create table ...")
    # Создание таблицы packets
    cur.execute("CREATE TABLE packets ("
                "[id] INTEGER PRIMARY KEY,"
                "[ip_source] TEXT,"
                "[ip_destination] TEXT,"
                "[port_source] INTEGER,"
                "[port_destination] INTEGER,"
                "[timestamp] FLOAT,"
                "[size] LONG)")

    # Получение массива пакетов
    packets = read_pcap(file_name)
    packets_size = len(packets)

    print("Fill database: ")
    valid_packets = []
    step = packets_size / 20
    bound = step
    for iter, buff in enumerate(packets):
        if iter >= bound:
            bound += step
            printProgressBar(iter, packets_size, length=40)
        try:
            eth = dpkt.ethernet.Ethernet(buff[1])
        except: continue
        # Если нет протокола IP
        if eth.type != dpkt.ethernet.ETH_TYPE_IP:
            continue
        ip_header = eth.data
        # Если нет протоколов TCP/UDP
        if ip_header.p not in (dpkt.ip.IP_PROTO_TCP, dpkt.ip.IP_PROTO_UDP):
            continue
        ip_s = socket.inet_ntoa(ip_header.src)
        ip_d = socket.inet_ntoa(ip_header.dst)
        port_s = ip_header.data.sport
        port_d = ip_header.data.dport
        timestamp = buff[0] - packets[0][0]
        size = len(buff[1])
        # Добавление в массив правильный пакет
        valid_packets.append(
            [iter + 1, ip_s, ip_d, port_s, port_d, timestamp, size])
    # Вставка всего массива valid_packets в таблицу packets
    cur.executemany(
        "INSERT INTO packets VALUES (?,?,?,?,?,?,?)", valid_packets)
    printProgressBar(1, 1, length=40, printEnd="\n")
    cur.close()
    con.commit()

def get_all_rows(db_name, table_name):
    con = connect(db_name)
    cur = con.cursor()
    rows = cur.execute("SELECT * FROM " + table_name).fetchall()
    cur.close()
    con.commit()
    return rows

def clear_db(db_name):
    print("Clear database ...")
    packets = get_all_rows(db_name, "packets")
    with open(db_name, 'w'): pass
    con = connect(db_name)
    cur = con.cursor()
    # Создание таблицы packets
    cur.execute("CREATE TABLE packets ("
                "[id] INTEGER PRIMARY KEY,"
                "[ip_source] TEXT,"
                "[ip_destination] TEXT,"
                "[port_source] INTEGER,"
                "[port_destination] INTEGER,"
                "[timestamp] FLOAT,"
                "[size] LONG)")
    cur.executemany("INSERT INTO packets VALUES (?,?,?,?,?,?,?)", packets)
    cur.close()
    con.commit()

# Функция удаления таблиц из баззы данных по переданному паттерну
def delete_tables(db_name, pattern):
    # Подключение к БД
    con = connect(db_name)
    cur = con.cursor()
    # Получение строк команд вида: DROP TABLE %pattern%
    commands = cur.execute(
        "SELECT 'DROP TABLE ' || name FROM sqlite_master WHERE name LIKE '" + pattern + "'").fetchall()
    size = len(commands)
    if size == 0:
        cur.close()
        con.commit()
        return
    print("Deleting tables:")
    step = size / 20
    bound = step
    for index, command in enumerate(commands):
        if index >= bound:
            bound += step
            printProgressBar(index + 1, size, length=40)
        # Выполнение команд вида: DROP TABLE %pattern%
        cur.execute(command[0])
    printProgressBar(1, 1, length=40, printEnd="\n")
    cur.close()
    con.commit()

# Функция выделения потоков
def select_streams(db_name, packets_limit):
    # Удаление всех таблиц начинающих с stream
    # delete_tables(db_name, "stream%")
    clear_db(db_name)
    # Подключение к БД
    con = connect(db_name)
    cur = con.cursor()
    # Создание таблицы streams для информации о потоках
    cur.execute("CREATE TABLE streams ("
                "[id] INTEGER PRIMARY KEY,"
                "[ip_source] TEXT,"
                "[ip_destination] TEXT,"
                "[port_source] INTEGER,"
                "[port_destination] INTEGER)")

    print("Selecting streams:")
    # Получение всех пакетов в отсортированном виде по IP и портам отправителя/получателя
    packets = cur.execute(
        "SELECT * FROM packets ORDER BY ip_source, ip_destination, port_source, port_destination").fetchall()
    packets_size = len(packets)

    stream_packets = 0
    count_threads = 0
    prev_ip_s, prev_ip_d, prev_p_s, prev_p_d = None, None, None, None
    prev_packets = []
    step = packets_size / 20
    bound = step
    for iter, packet in enumerate(packets):
        if iter >= bound:
            bound += step
            printProgressBar(iter + 1, packets_size, length=40)
        id, ip_s, ip_d, p_s, p_d, time, size = packet
        # Если IP и порты предыдущего пакета пакета совпадает с рассматрвиаемым, значит
        # это 1 поток
        if prev_ip_s == ip_s and prev_ip_d == ip_d and prev_p_s == p_s and prev_p_d == p_d:
            stream_packets += 1
            # Добавление пакета в массив пакетов потока
            prev_packets.append([id, time, size])
        # Иначе, мы перешли на другой поток
        else:
            # Если кол-во пакетов в предыдущем потока проходит по ограничению
            if stream_packets >= packets_limit:
                count_threads += 1
                # Вставляем в таблицу streams строчку с ID, IP и портом получателя/отправителя потока
                cur.execute("INSERT INTO streams VALUES({}, '{}', '{}', {}, {})"
                            .format(count_threads, prev_ip_s, prev_ip_d, prev_p_s, prev_p_d))
                # Создание отдельной таблицы вида: stream%ID_stream%
                cur.execute(f"CREATE TABLE stream{count_threads} ("
                            "[id] INTEGER PRIMARY KEY,"
                            "[timestamp] FLOAT,"
                            "[size] LONG)")
                # Вставка пакетов в отдельную таблицу потока
                cur.executemany(
                    f"INSERT INTO stream{count_threads} VALUES (?,?,?)", prev_packets)
            # Т.к. начался новый поток, то доабвляем в массив рассматрвиаемый пакет, и сохраняем
            # IP и порты этого пакета
            stream_packets = 1
            prev_packets = [[id, time, size]]
            prev_ip_s, prev_ip_d, prev_p_s, prev_p_d = ip_s, ip_d, p_s, p_d
    printProgressBar(1, 1, length=40, printEnd="\n")
    cur.close()
    con.commit()

# Функция подсчета прямых потоков
def get_counted_streams(db_cursor):
    # Получения всех потоков в отсортированном виде по IP отправителя и получателя
    all_streams = db_cursor.execute("SELECT DISTINCT ip_source, ip_destination, port_source, port_destination "
                                    "FROM packets ORDER BY ip_source, ip_destination").fetchall()
    all_streams_size = len(all_streams)
    prev_ip_s, prev_ip_d = all_streams[0][0], all_streams[0][1]
    count = 1
    counted_streams = []
    for i in range(1, all_streams_size):
        ip_s, ip_d = all_streams[i][0], all_streams[i][1]
        # Если IP получателя и отправителя совпадают с предыдущим потоком
        if ip_s == prev_ip_s and ip_d == prev_ip_d:
            count += 1
        else:
            # Добавление потока с количество прямых ему потоков + 1
            counted_streams.append([prev_ip_s, prev_ip_d, count])
            prev_ip_s, prev_ip_d = ip_s, ip_d
            count = 1
    return counted_streams

# Получение статистики потока, по его пакетам
def get_statistic(packets):
    packets_size = len(packets)
    full_size = 0
    full_time = packets[packets_size - 1][0] - packets[0][0]
    for packet in packets:
        full_size += packet[1]
    avg_size = full_size / packets_size
    avg_time = full_time / packets_size
    msd_size = msd_time = 0.0
    for packet in packets:
        msd_size += (avg_size - packet[1]) ** 2
        msd_time += (avg_time - packet[0]) ** 2
    msd_size = sqrt(msd_size / packets_size)
    msd_time = sqrt(msd_time / packets_size)
    # Кол-во пакетов, размер всех пакетов, средний размер пакетов, среднеквадратичное отклонение размеров
    # время потока, среднее время потока, среднеквадратичное отклонение времени
    return [packets_size, full_size, avg_size, msd_size, full_time, avg_time, msd_time]

# Функция добавления колонок со статистикой в таблицу потоков
def append_statistics(db_name):
    # Подключение к БД
    con = connect(db_name)
    cur = con.cursor()

    # Добавление колонок, если их нет
    for col_name, col_type in [
        ["count_packets", "INTEGER"],
        ["full_size", "LONG"],
        ["avg_size", "FLOAT"],
        ["msd_size", "FLOAT"],
        ["full_time", "FLOAT"],
        ["avg_time", "FLOAT"],
        ["msd_time", "FLOAT"],
        ["dir_streams", "INTEGER"],
            ["rev_streams", "INTEGER"]]:
        try:
            cur.execute(
                f"ALTER TABLE streams ADD COLUMN {col_name} {col_type}")
        except:
            pass

    print("Getting statistics:")
    # Получение массива потоков и кол-во их
    count_streams = get_counted_streams(cur)
    # Получение всех отобранных потоков
    streams = cur.execute("SELECT * FROM streams").fetchall()
    statistics = []

    streams_size = len(streams)
    step = streams_size / 20
    bound = step
    for stream in streams:
        if stream[0] >= bound:
            bound += step
            printProgressBar(stream[0], streams_size, length=40)
        # Получение временных меток и размеров пакетов рассматриваемого потока
        packets = cur.execute(
            f"SELECT timestamp, size FROM stream{stream[0]}").fetchall()

        # Подсчет кол-ва прямых и обратных потоков
        count_dir_streams = count_rev_streams = 0
        for cstream in count_streams:
            if stream[1] == cstream[0] and stream[2] == cstream[1]:
                count_dir_streams = cstream[2] - 1
            if stream[2] == cstream[0] and stream[1] == cstream[1]:
                count_rev_streams = cstream[2]

        # Получаем статистику потока
        static = get_statistic(packets)
        # Добавляем к статистике кол-во прямых и обратных потоков, и ID потока
        static.extend([count_dir_streams, count_rev_streams, stream[0]])
        # Добавляем статистику в массив статистик
        statistics.append(static)

    printProgressBar(1, 1, length=40, printEnd="\n")

    # Обновление значений колонок по ID потоков
    cur.executemany("UPDATE streams SET count_packets=?, full_size=?, avg_size=?, msd_size=?, "
                    "full_time=?, avg_time=?, msd_time=?, dir_streams=?, rev_streams=? WHERE id=?", statistics)
    cur.close()
    con.commit()

def print_statistic(db_name, rows_limit=50):
    con = connect(db_name)
    cur = con.cursor()

    print("Stream statistics:")
    statistics = cur.execute("SELECT id,count_packets,full_size,avg_size,msd_size,"
                             "full_time,avg_time,msd_time,dir_streams,rev_streams FROM streams").fetchall()

    table = PrettyTable(["stream", "count packets", "size", "avg size", "msd size",
                         "time", "avg time", "msd time", "direct stream", "reverse stream"])
    for i, st in enumerate(statistics):
        table.add_row([f"stream{st[0]}", st[1], st[2], f"{st[3]:.2f}",
                      f"{st[4]:.2f}", f"{st[5]:.3f}", f"{st[6]:.3f}", f"{st[7]:.2f}", st[8], st[9]])
        if i + 1 == rows_limit:
            break
    print(table)

    if i + 1 < len(statistics):
        print("Hidden", len(statistics) - i - 1, "threads")
    print("Count all threads: ", len(statistics))

    cur.close()
    con.close()
