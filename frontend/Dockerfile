# same Python as Colab
FROM python:3.13-slim

# working directory
WORKDIR /app

# copy the app files
COPY . .

# install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Streamlit port
EXPOSE 8501

# run Streamlit, XSRF off so CSV uploads work in the Codespace
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
