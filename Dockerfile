# Use a minimal Python 3.9 image
FROM python:3.11-slim-buster

# Create and set the working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy German model
RUN python -m spacy download de_core_news_sm

# Copy the rest of the project files
COPY . /app

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
