import csv

def load_words(csv_path):
    words = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("word")
            if word and word.isalpha():
                words.append(word.upper())
    return words

def get_word_list_by_length(word_list, length):
    return [w for w in word_list if len(w) == length]