FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

RUN pip install fairseq

WORKDIR /project/models

COPY . /project

