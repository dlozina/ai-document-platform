FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Download spaCy models
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download hr_core_news_sm
RUN python -m spacy download hr_core_news_lg

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8004"]
