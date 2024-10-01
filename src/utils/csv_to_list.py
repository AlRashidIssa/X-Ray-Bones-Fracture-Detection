import csv

def csv_to_list(csv_file_path):
    data = []
    with open(csv_file_path, newline='', model='r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data