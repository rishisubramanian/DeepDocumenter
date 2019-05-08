#!/usr/bin/env python

import os
from collections import defaultdict
import pandas as pd
from sys import argv, stdout
from tqdm import tqdm

def get_file_lists(rootdir, pbar=None):
    ext_counts = defaultdict(int)
    ext_sizes = defaultdict(int)
    for dirname, _, filelist in os.walk(rootdir):
        for f in filelist:
            full_path = os.path.abspath(os.path.join(dirname, f))
            if os.path.isfile(full_path):
                _, ext = os.path.splitext(f)
                ext_counts[ext] += 1
                filesize = os.path.getsize(full_path)
                ext_sizes[ext] += filesize
                if pbar is not None:
                    pbar.update(filesize)

    return ext_counts, ext_sizes

def merge_dicts_to_table(dict1, dict2):
    get_elem = lambda key: (key, dict1[key], dict2[key])
    merged_rows = map(get_elem, set(dict1.keys()) | set(dict2.keys()))
    return pd.DataFrame(merged_rows, columns=["ext", "count", "size"])

if __name__ == "__main__":
    pbar = tqdm(unit="B", unit_scale=True)
    counts, sizes = get_file_lists(argv[1], pbar)
    data = merge_dicts_to_table(counts, sizes)
    data.to_csv(stdout, index=False)
