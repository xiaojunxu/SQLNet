# Use an appropriate base image
FROM python:2.7

# Set the working directory inside the container
WORKDIR /app

# Copy the project code into the container
COPY . /app

RUN tar -xjvf data.tar.bz2

# Install dependencies
RUN pip install torch torchvision
RUN pip install -r requirements.txt
RUN pip install future

# Download and extract the glove embedding
RUN bash download_glove.sh
RUN python extract_vocab.py

# Expose any necessary ports
# EXPOSE <port_number>

# Set the command to run when the container starts
CMD ["python", "train.py"]