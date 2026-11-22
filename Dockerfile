FROM nvidia/cuda:13.4.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_ROOT=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends build-essential python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY filtros/ filtros/
COPY servicio/ servicio/
COPY herramientas/ herramientas/
COPY pruebas/ pruebas/
COPY wsgi.py gunicorn.conf.py ./

ARG GIT_COMMIT=desconocido
LABEL org.opencontainers.image.revision=$GIT_COMMIT
ENV GIT_COMMIT=$GIT_COMMIT

RUN useradd --create-home --uid 10001 app && mkdir -p /tmp/resultados && chown app /tmp/resultados
USER app
ENV DIR_RESULTADOS=/tmp/resultados PORT=5000

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/api/salud').status==200 else 1)"

CMD ["gunicorn", "-c", "gunicorn.conf.py", "wsgi:app"]
