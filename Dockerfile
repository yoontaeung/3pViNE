FROM rust:1-bookworm AS circom-builder

ARG CIRCOM_VERSION=v2.2.3

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${CIRCOM_VERSION} https://github.com/iden3/circom.git /tmp/circom
RUN cd /tmp/circom && cargo build --release

FROM node:20-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=circom-builder /tmp/circom/target/release/circom /usr/local/bin/circom

COPY package.json /opt/zk-vne-js/package.json
RUN cd /opt/zk-vne-js \
    && npm install \
    && npm install -g snarkjs@0.7.5

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/zk-vne-venv \
    && /opt/zk-vne-venv/bin/pip install --upgrade pip \
    && /opt/zk-vne-venv/bin/pip install -r /tmp/requirements.txt

ENV NODE_PATH=/opt/zk-vne-js/node_modules
ENV PATH=/opt/zk-vne-venv/bin:$PATH

WORKDIR /workspace
