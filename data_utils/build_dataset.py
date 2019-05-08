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
    for dirname, dirlist, filelist in os.walk(rootdir):
        for f in filelist:
            _, ext = os.path.splitext(f)
            if ext == ".py":
                yield os.path.abspath(os.path.join(dirname, f))

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
    if not docstring is None:
        ast_node.body = ast_node.body[1:]
    fn_text = astor.to_source(ast_node)
    # buffered_text = BytesIO(bytes(fn_text, "utf-8"))
    # tokens = list(tokenize(buffered_text.readline))
    return (
        # ast_node, 
        # tokens, 
        fn_text, 
        docstring
    )

def get_fns_from_file(filename, rootdir):
    src_tree = astor.parse_file(filename)
    rel_path = os.path.relpath(filename, start=rootdir)

    output_list = []

    class GetFunctions(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            output_list.append(extract_docstring(node))

    GetFunctions().visit(src_tree)
    output_list = [entry.insert(0, rel_path) for entry in output_list]
    return output_list

def main(rootdir):
    pool = mp.pool.Pool(12)

    progress_bar = tqdm()
    pbar_update = lambda *a: progress_bar.update()

    map_result = pool.map_async(get_fns_from_file, walkdir_py(rootdir), callback=pbar_update)

    all_fns = map_result.get()

    fn_entries = [entry for sublist in all_fns for entry in sublist]

    function_data = pd.DataFrame(fn_entries, columns=[
        "filename",
        # "ast",
        # "tokens",
        "fn_text",
        "docstring"
    ])
    function_data.to_pickle("fn_data.pickle.gz")

if __name__ == "__main__":
    rootdir = argv[1]
    main(rootdir)