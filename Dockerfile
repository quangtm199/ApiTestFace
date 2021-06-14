FROM tiangolo/uvicorn-gunicorn-fastapi:python3.6
RUN mkdir /fastapi
COPY requirements.txt /fastapi
WORKDIR /fastapi
RUN pip install --upgrade pip
RUN pip install -r requirements.txt 
COPY . /fastapi/
EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]