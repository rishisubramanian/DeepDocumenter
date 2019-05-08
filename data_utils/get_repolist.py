import pandas as pd
from tqdm import tqdm
import swifter
from operator import add
from functools import reduce

repo_list: pd.DataFrame = pd.read_csv("./repolist.csv")

repo_list.dropna(axis=0, how="any", subset=["langs"], inplace=True)
repo_list.drop([
    "siva_file",
    "file_count", 
    "commits_count", 
    "branches_count", 
    "fork_count", 
    "langs_byte_count", 
    "langs_lines_count", 
    "empty_lines_count", 
    "code_lines_count", 
    "comment_lines_count"
], axis=1, inplace=True)

for column in tqdm(repo_list.select_dtypes(include="object").columns[1:]):
    repo_list[column] = repo_list[column].swifter.progress_bar(True).apply(lambda x: x.split(","))

for column in tqdm(["langs_files_count"]):
    counts_dicts = repo_list[["langs", column]].swifter.progress_bar(True).apply(lambda x: dict(zip(x["langs"], x[column])), axis=1)
    counts_df = counts_dicts.swifter.progress_bar(True).apply(pd.Series).add_prefix(column + "_")
    counts_df.fillna(0, axis=1, inplace=True)
    repo_list = pd.concat([repo_list, counts_df], axis=1)
    repo_list.drop(column, axis=1, inplace=True)

repo_list.drop(["langs"], axis=1, inplace=True)

repo_list.to_csv("repolist_new.csv", index=False)

python_df = repo_list[["url", "langs_files_count_Python"]]
python_df = python_df.loc[pd.to_numeric(python_df["langs_files_count_Python"], errors="coerce") >= 10]

python_df.to_csv("python_repos.csv", index=False)

java_df = repo_list[["url", "langs_files_count_Java"]]
java_df = java_df.loc[pd.to_numeric(java_df["langs_files_count_Java"], errors="coerce") >= 10]

java_df.to_csv("java_repos.csv", index=False)