import pandas as pd
from tqdm import tqdm
from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException
from io import BytesIO
from tokenize import tokenize
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
# tqdm.pandas()

pbar = ProgressBar()
pbar.register()

dataset = pd.read_pickle("fn_data.pickle.gz")
dataset = dd.from_pandas(dataset, npartitions=12)
# dataset = dataset.loc[dataset[2].str.len() > 10]
# dataset = dataset.loc[dataset[2].str.contains(r"[A-Za-z]", regex=True)]


def detect_language(string):
    try:
        langs = detect(string)
        return langs
    except LangDetectException:
        return None


dataset["languages"] = dataset["docstring"].apply(detect_language, meta=pd.Series(dtype="str", name="languages"))
# dataset["languages"] = dataset["docstring"].apply(detect_language)

dataset = dataset.loc[dataset["languages"] == "en"]


def get_tokens(fn_string):
    fn_bytes = BytesIO(bytes(fn_string, "utf-8"))
    tokens = list(tokenize(fn_bytes.readline))
    return tokens


dataset = dataset.compute(scheduler="processes")

dataset.drop("languages", axis=1, inplace=True)

dataset.to_pickle("fn_data_cleaned.pickle.gz")
