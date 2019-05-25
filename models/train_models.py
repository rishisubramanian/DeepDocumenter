#!/usr/bin/env python3

import subprocess
import multiprocessing
import os
import json
import io

num_cpu = multiprocessing.cpu_count()

gpu_name = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode()
is_tesla = "tesla" in gpu_name.lower()

training_options = [
    "--log-interval", "100",
    "--log-format", "json",
    "--num-workers", str(num_cpu),
    "--skip-invalid-size-inputs-valid-test",
    "--max-epoch", "10",
    "--optimizer", "adam",
    "--learning-rate", "0.001",
    "--lr-scheduler", "fixed",
    "--save-interval-updates", "10000",
    "--keep-interval-updates", "1"
]

if is_tesla:
    training_options.append("--fp16")

seq2seq_model = [
    "--arch", "lstm",
    "--encoder-embed-dim", "256",
    "--decoder-embed-dim", "256",
    "--encoder-hidden-size", "256",
    "--decoder-hidden-size", "256",
    "--encoder-layers", "2",
    "--decoder-layers", "2",
    # "--encoder-bidirectional"
]

transformer_model = [
    "--arch", "transformer",
    "--encoder-layers", "2",
    "--decoder-layers", "2",
    "--encoder-embed-dim", "256",
    "--decoder-embed-dim", "256",
    "--encoder-ffn-embed-dim", "512",
    "--decoder-ffn-embed-dim", "512"
]

dynamicconv_model = [
    "--arch", "lightconv",
    "--encoder-embed-dim", "256",
    "--decoder-embed-dim", "256",
    "--encoder-ffn-embed-dim", "512",
    "--decoder-ffn-embed-dim", "512",
    "--encoder-layers", "2",
    "--decoder-layers", "2",
    "--encoder-conv-type", "dynamic",
    "--decoder-conv-type", "dynamic"
]

model_names = [
    # "seq2seq",
    "transformer"
    # "dynamicconv"
]

model_params = [
    # seq2seq_model,
    transformer_model
    # dynamicconv_model
]

datasets = [
    "limited_vocab",
    "bpe"
]

command_list = {}

for model_name, model in zip(model_names, model_params):
    for dataset in datasets:
        data_path = [os.path.join("../data", dataset)]
        checkpoint_dir = os.path.join("checkpoints", model_name, dataset)
        params = ["fairseq-train"] + data_path + training_options + model + ["--save-dir", checkpoint_dir]
        config_name = "_".join([model_name, dataset])
        command_list[config_name] = params
        # print(params)

for name, command in command_list.items():
    print(f"Training: {name}")
    with open(name + "_log.json", "wb") as log_file:
        training_proc = subprocess.run(command, stdout=log_file)
        # for line in io.TextIOWrapper(training_proc.stdout, encoding="utf-8"):
        #     print(line)
        #     if line.strip()[0] == "{":
        #         log_file.write(line)
        #         try:
        #             log_line = json.loads(line)
        #             print(f"Epoch: {log_line['Epoch']}\tUpdate: {log_line['Update']}")
        #         except json.JSONDecodeError:
        #             pass
