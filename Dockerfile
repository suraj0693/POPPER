FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY benchmark_scripts /app/benchmark_scripts

COPY popper /app/popper

COPY baseline_agents /app/baseline_agents

# Create directories for storing output and logs
RUN mkdir -p /app/data /app/.logs && chmod -R 777 /app/data /app/.logs

RUN useradd -m nonrootuser
USER nonrootuser

ENTRYPOINT ["python"]
