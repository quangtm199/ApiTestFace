FROM nvidia/cuda:10.2-base
CMD nvidia-smi
RUN apt-get update 
RUN apt-get -y install python3.6
RUN apt-get -y install python3-pip
RUN mkdir /fastapi
COPY requirements.txt /fastapi
WORKDIR /fastapi
RUN pip install --upgrade pip
RUN pip install uvicorn
RUN pip install mxnet
RUN pip install pandas
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt 
COPY . /fastapi/
EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80","--reload"]
