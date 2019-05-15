import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import spacy
from io import BytesIO
from tokenize import tokenize
from multiprocess import Pool

dataset: pd.DataFrame = pd.read_pickle("data/fn_data_cleaned.pickle.gz")
dataset.drop("filename", axis=1, inplace=True)
dataset = shuffle(dataset)

nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])

def tokenize_python(fn_string):
    token_buffer = BytesIO(bytes(fn_string, "utf-8"))
    tokens = list(tokenize(token_buffer.readline))
    encoded_tokens = [token.string.encode("unicode_escape") for token in tokens]
    return b"\t".join(encoded_tokens)

def tokenize_docstring(docstring):
    doc = nlp(docstring)
    return b"\t".join([token.text.encode("unicode_escape") for token in doc])

def tokenize_row(row):
    tokenized_python = tokenize_python(row.fn_body)
    tokenized_docstring = tokenize_docstring(row.docstring)

    return tokenized_python, tokenized_docstring

workers = Pool(12)
results = workers.map(tokenize_row, dataset.itertuples(index=False))
workers.close()
workers.join()

tokenized_python, tokenized_docstring = zip(*results)
del results

with open("data/python_all_tokens.txt", "wb") as python_file:
    python_file.write(b"\n".join(tokenized_python))

with open("data/python_all_docstrings.txt", "wb") as text_file:
    text_file.write(b"\n".join(tokenized_docstring))

num_lines = len(tokenized_python)

train_prop, val_prop = 0.8, 0.1

train_end = int(train_prop * num_lines)
val_end = int((train_prop + val_prop) * num_lines)

train_python_lines = tokenized_python[:train_end]
train_doc_lines = tokenized_docstring[:train_end]

with open("data/train.py", "wb") as python_file:
    python_file.write(b"\n".join(train_python_lines))

with open("data/train.en", "wb") as text_file:
    text_file.write(b"\n".join(train_doc_lines))

val_python_lines = tokenized_python[train_end:val_end]
val_doc_lines = tokenized_docstring[train_end:val_end]

with open("data/valid.py", "wb") as python_file:
    python_file.write(b"\n".join(val_python_lines))

with open("data/valid.en", "wb") as text_file:
    text_file.write(b"\n".join(val_doc_lines))

test_python_lines = tokenized_python[val_end:]
test_doc_lines = tokenized_docstring[val_end:]

with open("data/test.py", "wb") as python_file:
    python_file.write(b"\n".join(test_python_lines))

with open("data/test.en", "wb") as text_file:
    text_file.write(b"\n".join(test_doc_lines))