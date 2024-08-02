# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir pandas prometheus_client

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run export_metrics.py when the container launches
CMD ["python", "export_metrics.py"]