FROM nvidia/cuda:10.0-base-ubuntu16.04
CMD nvidia-smi
RUN apt-get update 
RUN apt-get install unzip
RUN apt-get install python3
RUN apt-get install python3-pip

RUN mkdir /fastapi
COPY requirements.txt /fastapi
WORKDIR /fastapi
RUN pip3 install --upgrade pip
RUN pip3 install uvicorn
RUN pip3 install mxnet
RUN pip3 install pandas
RUN apt-get update ##[editead]
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt 
COPY . /fastapi/
EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80","--reload"]
