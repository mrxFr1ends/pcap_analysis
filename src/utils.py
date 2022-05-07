import time
from prettytable import PrettyTable

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)

def elapsed_time(func, *params, accuracy=3):
    start_time = time.monotonic()
    func(*params)
    end_time = time.monotonic() - start_time
    print("Elapsed time: {:{}} seconds".format(end_time, "." + str(accuracy) + "f"))
    return end_time

def print_table(rows, labels, rows_limit=0, float_format='.2', header=True, border=True):
    table = PrettyTable(labels)
    for iter, row in enumerate(rows):
        table.add_row(row)
        if iter + 1 == rows_limit:
            break
    table.float_format = float_format
    print(table.get_string(header=header, border=border))
    if iter + 1 < len(rows):
        print("Hidden", len(rows) - iter - 1, "rows")
