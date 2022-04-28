# Пример запуска программы

```
python practice.py -pn PCAP_NAME 
```

После запуска программы, будет создан файл `database` с расширением `.db`,
по умолчанию потоки будут содержать не меньше `100` пакетов, во время программы будут
показаны окна с метриками и т.д., так же в папке output появятся картинки с показанными
метриками и т.д.

# Подбор лимита по пакетам

Для удобного подборка лимита по пакетам, после первого запуска программы, 
можно не указывать аргумент `-pn/--pcap_name`, а указывать флаг `--no-create` и аргумент `-l/--limit`

Например:
```
python practice.py --no-create -l LIMIT
```

Запущенная программа будет использовать потоки из ранее созданного файла с расширением `.db`

# Вывод программы с аргументом `--help`

```
usage: practice.py [-h] [-dn DB_NAME] [-pn PCAP_NAME] [-l LIMIT] [--no-create] [--no-show] [--no-save]

options:
  -h, --help                            show this help message and exit
  -dn DB_NAME, --db_name DB_NAME        Database filename
  -pn PCAP_NAME, --pcap_name PCAP_NAME  Pcap filename
  -l LIMIT, --limit LIMIT               Min count packets in thread
  --no-create                           Dont create new database file
  --no-show                             Dont show windows
  --no-save                             Dont save windows
```