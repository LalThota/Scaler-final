FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Expose port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
