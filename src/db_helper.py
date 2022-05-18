import dpkt
from sqlite3 import *
from .utils import print_table, printProgressBar
import socket
from math import sqrt

# Функция парсинга pcap файла, возвращает массив прочитанных пакетов
def read_pcap(file_name):
    print("Read pcap file ...")
    # Открываем файл file_name для чтения в бинарном режиме
    with open(file_name, 'rb') as file:
        # Возвращаем массив всех пакетов из файла file_name
        return dpkt.pcap.Reader(file).readpkts()

# Функция создания базы данных из файла pcap
def pcap2db(file_name, db_name="database.db"):
    # Очистка файла
    with open(db_name, 'w'): pass
    # Подключаемся к базе данных
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
    # Массив валидных пакетов
    valid_packets = []
    # Шаг обновления индикатора прогресса
    step = packets_size / 20
    # Граница индикатора прогресса
    bound = step
    for iter, buff in enumerate(packets):
        # Если заходим за границу индикатора, обновляем индикатор
        # меняя границу на шаг обновления
        if iter >= bound:
            bound += step
            printProgressBar(iter, packets_size, length=40)

        # Пробуем получить необходимую информацию о пакете
        try:
            # Получаем данные об Ethernet протоколе
            eth = dpkt.ethernet.Ethernet(buff[1])
            # Получаем данные об IP протоколе
            ip_header = eth.data
            # Получаем необходимую информацию о пакете
            ip_source = socket.inet_ntoa(ip_header.src)
            ip_destination = socket.inet_ntoa(ip_header.dst)
            port_source = ip_header.data.sport
            port_destination = ip_header.data.dport
            timestamp = buff[0] - packets[0][0]
            size = len(buff[1])
            # Если при получении информации о пакете, не возникло ошибок
            # добавляем пакет в массив валидных пакетов
            valid_packets.append([iter + 1,
                                ip_source,
                                ip_destination,
                                port_source,
                                port_destination,
                                timestamp,
                                size])
        # Если по каким-либо причинам возникла ошибка, пропускаем пакет
        except: continue
    
    # Вставка всего массива valid_packets в таблицу packets
    cur.executemany(
        "INSERT INTO packets VALUES (?,?,?,?,?,?,?)", valid_packets)
    printProgressBar(1, 1, length=40, printEnd="\n")
    cur.close()
    con.commit()

# Функция получения всех строк из таблицы с названием table_name
def get_all_rows(db_name, table_name):
    # Подключаемся к базе данных
    con = connect(db_name)
    cur = con.cursor()
    # Получаем все строчки со всеми столбцами из таблицы table_name
    rows = cur.execute("SELECT * FROM " + table_name).fetchall()
    cur.close()
    con.commit()
    return rows

# Функция очищение базы данных от всех таблиц кроме packets
def clear_db(db_name):
    print("Clear database ...")
    # Получаем все пакеты из таблицы packets
    packets = get_all_rows(db_name, "packets")
    # Очищаем файл
    with open(db_name, 'w'): pass
    # Подключаемся к базе данных
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
    # Записываем обратно все пакеты в таблицу packets
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
    # Очищаем базу данных
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
    query = "SELECT * FROM packets ORDER BY ip_source, ip_destination, port_source, port_destination"
    packets = cur.execute(query).fetchall()
    packets_size = len(packets)

    # Кол-во пакетов в потоке
    stream_packets = 0
    # Кол-во потоков
    count_streams = 0
    # Предыдущие значения IP/портов отправителя/получателя
    prev_ip_s, prev_ip_d, prev_p_s, prev_p_d = None, None, None, None
    # Пакеты предыдущего потока
    prev_packets = []
    # Шаг для обновления индикатора прогресса
    step = packets_size / 20
    # Граница индикатора
    bound = step
    for iter, packet in enumerate(packets):
        # Если заходим за границу индикатора, обновляем индикатор
        # меняя границу на шаг обновления
        if iter >= bound:
            bound += step
            printProgressBar(iter + 1, packets_size, length=40)
        # Переменные с информацией о пакете
        id, ip_s, ip_d, p_s, p_d, time, size = packet
        # Если IP и порт предыдущего пакета совпадает с рассматрвиаемым, значит
        # это один поток
        if prev_ip_s == ip_s and prev_ip_d == ip_d and prev_p_s == p_s and prev_p_d == p_d:
            stream_packets += 1
            # Добавление пакета в массив пакетов потока
            prev_packets.append([id, time, size])
        # Иначе, мы перешли на другой поток
        else:
            # Если кол-во пакетов в предыдущем потоке проходит по ограничению
            if stream_packets >= packets_limit:
                count_streams += 1
                # Вставляем в таблицу streams строчку с ID, IP и портом получателя/отправителя потока
                cur.execute("INSERT INTO streams VALUES({}, '{}', '{}', {}, {})"
                            .format(count_streams, prev_ip_s, prev_ip_d, prev_p_s, prev_p_d))
                # Создание отдельной таблицы вида: stream%ID_stream%
                cur.execute(f"CREATE TABLE stream{count_streams} ("
                            "[id] INTEGER PRIMARY KEY,"
                            "[timestamp] FLOAT,"
                            "[size] LONG)")
                # Вставка пакетов в отдельную таблицу потока
                cur.executemany(
                    f"INSERT INTO stream{count_streams} VALUES (?,?,?)", prev_packets)
            # Т.к. начался новый поток, то доабвляем в массив рассматрвиаемый пакет, и сохраняем
            # IP и порты этого потока
            stream_packets = 1
            prev_packets = [[id, time, size]]
            prev_ip_s, prev_ip_d, prev_p_s, prev_p_d = ip_s, ip_d, p_s, p_d
    printProgressBar(1, 1, length=40, printEnd="\n")
    cur.close()
    con.commit()

# Функция получения кол-ва прямых потоков для каждого IP получателя/отправителя
def get_counted_streams(db_cursor):
    # Получения всех пакетов в отсортированном виде по IP получателя/отправителя
    query = '''
    SELECT DISTINCT ip_source, ip_destination, port_source, port_destination 
    FROM packets ORDER BY ip_source, ip_destination
    '''
    all_streams = db_cursor.execute(query).fetchall()
    all_streams_size = len(all_streams)
    # IP получателя/отправителя предыдущего потока
    prev_ip_s, prev_ip_d = all_streams[0][0], all_streams[0][1]
    # Кол-во прямых потоков
    count = 1
    # Массив с IP получателя/отправителя и кол-вом прямых потоков
    counted_streams = []
    for i in range(1, all_streams_size):
        # Переменные с IP получателя/отправителя
        ip_s, ip_d = all_streams[i][0], all_streams[i][1]
        # Если IP получателя и отправителя совпадают с предыдущими
        # то мы все еще рассматриваем один поток
        if ip_s == prev_ip_s and ip_d == prev_ip_d:
            count += 1
        # Иначе если мы перешли на новый поток
        else:
            # то добавляем в массив информацию о предыдущем потоке
            counted_streams.append([prev_ip_s, prev_ip_d, count])
            prev_ip_s, prev_ip_d = ip_s, ip_d
            count = 1
    return counted_streams

# Получение статистики потока, по его пакетам
def get_statistic(packets):
    # Кол-во пакетов
    packets_size = len(packets)
    # Размер всех пакетов
    full_size = 0
    for packet in packets:
        full_size += packet[1]
    # Начало потока
    start_time = packets[0][0]
    # Конец потока
    end_time = packets[packets_size - 1][0]
    # Длительность потока
    full_time = end_time - start_time
    # Средний размер пакета
    avg_size = full_size / packets_size
    # Среднее время пакета
    avg_time = full_time / packets_size
    # Максимальное среднеквадратичное отклонение
    # размера и времени
    msd_size = msd_time = 0.0
    for packet in packets:
        msd_size += (avg_size - packet[1]) ** 2
        msd_time += (avg_time - packet[0]) ** 2
    msd_size = sqrt(msd_size / packets_size)
    msd_time = sqrt(msd_time / packets_size)
    # Кол-во пакетов, размер всех пакетов, средний размер пакетов, среднеквадратичное отклонение размеров
    # время потока, среднее время потока, среднеквадратичное отклонение времени
    return [start_time, end_time, packets_size,
            full_size, avg_size, msd_size, 
            full_time, avg_time, msd_time]

# Функция добавления колонок со статистикой в таблицу потоков
def append_statistics(db_name):
    # Подключение к БД
    con = connect(db_name)
    cur = con.cursor()

    # Добавление колонок, если их нет
    for col_name, col_type in [
        ["start_time", "FLOAT"],
        ["end_time", "FLOAT"],
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
            query = f"ALTER TABLE streams ADD COLUMN {col_name} {col_type}"
            cur.execute(query)
        except:
            pass

    print("Getting statistics:")
    # Получение кол-ва прямых потоков для всех IP получателя/отправителя
    count_streams = get_counted_streams(cur)
    # Получение всех отобранных потоков
    streams = cur.execute("SELECT * FROM streams").fetchall()
    # Массив с статистикой каждого потока
    statistics = []
    # Кол-во потоков
    streams_size = len(streams)
    # Шаг для обновления индикатора прогресса
    step = streams_size / 20
    # Граница индикатора
    bound = step
    for stream in streams:
        # Если заходим за границу индикатора, обновляем индикатор
        # меняя границу на шаг обновления
        if stream[0] >= bound:
            bound += step
            printProgressBar(stream[0], streams_size, length=40)
        # Получение временных меток и размеров пакетов рассматриваемого потока
        query = f"SELECT timestamp, size FROM stream{stream[0]}"
        packets = cur.execute(query).fetchall()

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
    query = '''
    UPDATE streams SET start_time=?, end_time=?, count_packets=?, 
    full_size=?, avg_size=?, msd_size=?, 
    full_time=?, avg_time=?, msd_time=?, 
    dir_streams=?, rev_streams=? WHERE id=?
    '''
    cur.executemany(query, statistics)
    cur.close()
    con.commit()

def print_statistic(db_name, rows_limit=50):
    con = connect(db_name)
    cur = con.cursor()

    print("Stream statistics:")
    query = '''
    SELECT id,start_time,end_time,count_packets,full_size,avg_size,msd_size,
    full_time,avg_time,msd_time,dir_streams,rev_streams FROM streams
    '''
    statistics = cur.execute(query).fetchall()
    labels = ["stream", "start time", "end time", "count packets",
              "size", "avg size", "msd size",
              "time", "avg time", "msd time",
              "direct stream", "reverse stream"]
    print_table(statistics, labels, rows_limit=rows_limit)
    print("Count all threads: ", len(statistics))

    cur.close()
    con.commit()
