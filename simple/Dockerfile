FROM registry.access.redhat.com/ubi9/ubi-minimal:9.8

RUN microdnf -y --setopt=install_weak_deps=0 install python3.12 python3.12-pip util-linux \
    && microdnf clean all

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN python3.12 -m venv /opt/venv \
    && pip install -r requirements.txt

COPY solution.py .
RUN mkdir outputs && chown 1001:0 outputs

USER 1001
CMD ["python", "solution.py"]
