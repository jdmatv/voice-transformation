FROM python:3.9-slim-buster

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app/main.py .
COPY ./app/htmldirectory /code/htmldirectory/
COPY ./app/library /code/library/
EXPOSE 8080
RUN dir

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080"]