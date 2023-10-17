FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]


RUN pipenv install --system --deploy
COPY ["predict.py", "churn-model.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
# ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]