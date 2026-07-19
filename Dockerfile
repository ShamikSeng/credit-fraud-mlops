FROM python:3.10-slim-bookworm

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN pytest -v tests/

EXPOSE 8005

ENTRYPOINT ["python"]

CMD ["main.py"]
