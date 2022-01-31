FROM python:3.8-bullseye

COPY ./bomberland/requirements.txt /app/bomberland/requirements.txt
COPY . /app/
WORKDIR /app
RUN python -m pip install -r bomberland/requirements.txt
RUN python -m pip install -e .
# RUN python -m bomberland/unittest
ENTRYPOINT PYTHONUNBUFFERED=1 python bomberland/agent.py
