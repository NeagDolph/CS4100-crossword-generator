import csv

with open("/Users/kelseynihezagirwe/Desktop/CS4100-crossword-generator/nytcrosswords.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print("ROW:", row)
