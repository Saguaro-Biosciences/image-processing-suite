FROM python:3.13

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential

# Install pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Clone your repository
RUN git clone https://github.com/youruser/yourrepo.git /home/ubuntu/yourrepo

# Set working directory
WORKDIR /home/ubuntu/yourrepo

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on (if applicable)
#EXPOSE 5000

# Command to run your app (adjust accordingly)
#CMD ["python3", "your_script.py"]
