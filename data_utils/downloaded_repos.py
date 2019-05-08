from subprocess import run, CalledProcessError, DEVNULL, TimeoutExpired
import pandas as pd
from tqdm import tqdm
from shutil import rmtree
import multiprocessing as mp
import os
from urllib.parse import urlparse
from pathlib import PurePosixPath

repo_df = pd.read_csv("./python_java_repos.csv")

base_dir = os.path.expanduser("~/ext/repos")

os.chdir(base_dir)

def get_repo(url):
    """Function to download a repository and clean it by downloading its .git folder"""
    parsed_url = urlparse(url)
    repo_name = PurePosixPath(parsed_url.path).name
    ssh_url = "git@github.com:" + parsed_url.path.lstrip("/")

    if not os.path.exists(repo_name):
        try:
            run(
                ["git", "clone", "--depth", "1", "--quiet", ssh_url], 
                check=True, 
                stderr=DEVNULL, 
                stdout=DEVNULL
            )
            rmtree(os.path.join(repo_name, ".git"))
        except (CalledProcessError, TimeoutExpired, KeyboardInterrupt):
            # print(repo_name + " failed")
            if os.path.exists(repo_name):
                rmtree(repo_name)



progress_bar = tqdm(total=len(repo_df["url"]))
pbar_update = lambda *a: progress_bar.update()

pool = mp.Pool(12)

for i in repo_df["url"]:
    pool.apply_async(get_repo, args=(i,), callback=pbar_update)

pool.close()
pool.join()
