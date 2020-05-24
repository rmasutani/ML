FROM continuumio/anaconda3

RUN pip install --upgrade pip &&\
    pip install autopep8 && \
    pip install Keras &&\
    pip install Torch Torchvision

COPY ./workspace ./

WORKDIR /workspace

CMD jupyter-lab --no-browser \
    --port=8888 --ip=0.0.0.0 --allow-root