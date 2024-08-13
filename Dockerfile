FROM docker.io/dataloopai/dtlpy-agent:cpu.py3.10.opencv
USER root
RUN apt update && apt install -y curl gpg software-properties-common

USER 1000
WORKDIR /tmp
ENV HOME=/tmp
RUN pip install jira \
    "numpy<2.0.0" \
    dtlpy \
    python-dotenv \
    openai


# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/llm-tools:0.1.21 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/llm-tools:0.1.21

# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/llm-tools:0.1.20 bash
