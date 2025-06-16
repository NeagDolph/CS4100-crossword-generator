import csv

def get_word_list_by_length(word_list, length):
    return [w for w in word_list if len(w) == length]

def load_words(csv_path):
    words = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().upper()
            if word and word != "WORD":
                words.append(word)
    return words


