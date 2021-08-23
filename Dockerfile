FROM python:3.8.0-slim-buster
MAINTAINER Alice Pagnoux (apagnoux@cisco.com)

# Install linux packages
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y libgl1-mesa-glx ffmpeg wget
RUN apt-get update && apt-get -y install --no-install-recommends
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

# Install python dependencies
RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Run detection
CMD python3 detect.py

# ubuntu  = 4.85GB
# python:3.8.0-slim  = 4.92GB
# tensorflow/tensorflow:latest-py3  = 6.56GB