# Base image with the same Python version as Colab
FROM python:3.13-slim

# Working directory inside the container
WORKDIR /app

# Copy app.py, requirements.txt and the model into the image
COPY . .

# Install dependencies (no cache keeps the image small)
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# The API listens on port 7860
EXPOSE 7860

# Run with gunicorn (4 workers) on all interfaces; app:superkart_api = file app.py, object superkart_api
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:superkart_api"]
