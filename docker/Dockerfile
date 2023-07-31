FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3 python3-pip

RUN ln -s /usr/bin/python3.10 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

# Install pip
RUN apt-get install python3-pip

RUN apt-get install -y build-essential

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install Python dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /storage

WORKDIR /storage/speed_estimation
