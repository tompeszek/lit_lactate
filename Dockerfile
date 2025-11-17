# Use Python 3.12 base image
FROM python:3.12-slim

# Install system dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY uv.lock* ./
COPY app.py .
COPY lactate_analysis.py .
COPY training_zones.py .

# Install UV
RUN pip install uv

# Install Python dependencies
RUN uv sync --frozen

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
