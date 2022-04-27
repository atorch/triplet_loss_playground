FROM tensorflow/tensorflow:2.3.0-gpu

RUN apt-get update

WORKDIR /home/triplet_loss_playground

ADD requirements.txt .

RUN pip install -r requirements.txt

ENV TFDS_DATA_DIR=/home/triplet_loss_playground/tfds_data