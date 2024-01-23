FROM nvcr.io/nvidia/pytorch:23.05-py3

ARG USER_ID
ARG GROUP_ID
ARG USER
ARG DEBIAN_FRONTEND=noninteractive

RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos "" --uid $USER_ID --gid $GROUP_ID $USER

WORKDIR /nfs/home/$USER/AdapterSIS
EXPOSE 8888

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip3 install --upgrade -r requirements.txt
RUN pip install "opencv-python-headless<4.3"