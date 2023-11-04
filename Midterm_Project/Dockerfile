FROM python:3.10

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN pip install gunicorn

EXPOSE 9696
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
