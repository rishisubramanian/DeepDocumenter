#!/usr/bin/env python

import os
import ast
import astor
import copy
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
from sys import argv
from tokenize import tokenize
from io import BytesIO

def walkdir_py(rootdir):
    for dirname, _, filelist in os.walk(rootdir):
        for f in filelist:
            _, ext = os.path.splitext(f)
            if ext == ".py":
                yield os.path.abspath(os.path.join(dirname, f)), dirname

def get_file_contents(filename, rootdir):
    with open(filename, "r") as src_file:
        contents = src_file.read()
    rel_path = os.path.relpath(filename, start=rootdir)
    return (
        rel_path,
        contents
    )

def extract_docstring(ast_node):
    ast_node = copy.deepcopy(ast_node)
    docstring = ast.get_docstring(ast_node, clean=True)
    if docstring is None:
        return None
    ast_node.body = ast_node.body[1:]
    try:
        fn_text = astor.to_source(ast_node)
    except OverflowError:
        return None
    # buffered_text = BytesIO(bytes(fn_text, "utf-8"))
    # tokens = list(tokenize(buffered_text.readline))
    return [
        # ast_node, 
        fn_text, 
        docstring
        # tokens
    ]

def get_fns_from_file(filename, rootdir):
    try:
        src_tree = astor.parse_file(filename)
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError, MemoryError, ValueError):
        return []

    rel_path = os.path.relpath(filename, start=rootdir)

    output_list = []

    class GetFunctions(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            docstring_data = extract_docstring(node)
            if docstring_data is not None:
                output_list.append(docstring_data)

    try:
        GetFunctions().visit(src_tree)
    except RecursionError:
        return []
    output_list = [[filename] + entry for entry in output_list]

    return output_list

def main(rootdir):
    pool = mp.Pool(12)

    progress_bar = tqdm()
    pbar_update = lambda *a: progress_bar.update()

    results = []

    for filepath in walkdir_py(rootdir):
        results.append(pool.apply_async(get_fns_from_file, filepath, callback=pbar_update))

    results = [result.get() for result in results]
    fn_entries = [entry for sublist in results for entry in sublist]

    function_data = pd.DataFrame(fn_entries, columns=[
        "filename",
        "fn_body",
        "docstring"
        # "tokens"
    ])
    function_data.to_pickle("fn_data.pickle.gz")

if __name__ == "__main__":
    rootdir = argv[1]
    main(rootdir)