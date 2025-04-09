# Use an official Python runtime as a parent image
FROM python:3.13

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (Git, Docker)
RUN apt-get update && apt-get install -y \
    git \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to run when the container starts
CMD ["bash"]
