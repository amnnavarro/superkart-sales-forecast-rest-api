# Same Python version as Colab
FROM python:3.13-slim

# Working directory inside the container
WORKDIR /app

# Copy the app files into the image
COPY . .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Streamlit's port
EXPOSE 8501

# Run Streamlit on all interfaces; XSRF protection is off so CSV uploads work through the Codespace proxy
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
