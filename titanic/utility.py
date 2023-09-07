import datetime


def print_time():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("-" * 25 * 5 + f"{current_time}")
