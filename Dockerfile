FROM mambaorg/micromamba:latest

ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN micromamba install -y -n base -c conda-forge \
       python=3.10 \
       wget && \
    micromamba clean --all --yes

WORKDIR /app

COPY requirements.txt .

# Install pip dependencies into the micromamba environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY 12_lead_ECGFounder.pth .
COPY ukb/*.py .

USER root
