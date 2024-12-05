FROM python:3.12

COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt

ENV FLASK_APP=app.py

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
