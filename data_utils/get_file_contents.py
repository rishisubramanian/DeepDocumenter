import os
import multiprocessing as mp
import sqlite3
from sys import argv
from tqdm import tqdm

db_location = "/home/rishi/ext/file_data.sqlite"

def insert_python(filename, rootdir):
    rel_path = os.path.relpath(filename, start=rootdir)
    with open(filename, "r") as src_file:
        contents = src_file.read()

    db_conn = sqlite3.Connection(db_location)
    cursor = db_conn.cursor()
    cursor.execute("insert into python_files values (?, ?)", (rel_path, contents))
    db_conn.commit()
    db_conn.close()

def insert_java(filename, rootdir):
    rel_path = os.path.relpath(filename, start=rootdir)
    with open(filename, "r") as src_file:
        contents = src_file.read()

    db_conn = sqlite3.Connection(db_location)
    cursor = db_conn.cursor()
    cursor.execute("insert into java_files values (?, ?)", (rel_path, contents))
    db_conn.commit()
    db_conn.close()

pool = mp.Pool(12)

def walkdir_py(rootdir):
    progress_bar = tqdm()
    pbar_update = lambda *a: progress_bar.update()

    for dirname, dirlist, filelist in os.walk(rootdir):
        for f in filelist:
            _, ext = os.path.splitext(f)
            if ext == ".py":
                pool.apply_async(insert_python, (os.path.abspath(os.path.join(dirname, f)), dirname), callback=pbar_update)
            elif ext == ".java":
                pool.apply_async(insert_java, (os.path.abspath(os.path.join(dirname, f)), dirname), callback=pbar_update)

if __name__ == "__main__":
    walkdir_py(argv[-1])