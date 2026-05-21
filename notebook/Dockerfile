FROM eu.gcr.io/ramadhan-s4g/eo-base:latest

RUN apt-get upgrade & apt-get update

RUN apt-get install mkcert

RUN code-server \
  --install-extension ms-python.python \
  --install-extension ms-toolsai.jupyter \
  --install-extension charliermarsh.ruff

WORKDIR /usr/src/app

RUN mkcert localhost
RUN mkcert 0.0.0.0

COPY requirements.txt .

RUN python3 -m venv .venv && \
  .venv/bin/pip install -r requirements.txt

EXPOSE 8080

CMD [ "code-server", "--auth", "none", "--host", "0.0.0.0", "--port", "8080", "/usr/src/app", "--cert-key", "0.0.0.0.pem" ]
