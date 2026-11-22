# base image (same Python as Colab)
FROM python:3.13-slim

# working directory
WORKDIR /app

# copy the files into the image
COPY . .

# install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# port 7860
EXPOSE 7860

# run with gunicorn, 4 workers
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:superkart_api"]
