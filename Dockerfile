# Use the official Python image from the Docker Hub
FROM python:3.10 as poetry2requirements

# Copy the Python project files into the image
COPY pyproject.toml poetry.lock /

# Set the environment variable for the poetry installation directory
ENV POETRY_HOME=/etc/poetry

# Install Poetry package manager
RUN pip3 install poetry

# Set the working directory inside the container
WORKDIR /app

# Export the dependencies from Poetry to a requirements.txt file
RUN python3 -m poetry export --without-hashes -f requirements.txt -o requirements.txt

# Install R and Rscript
RUN apt-get update && \
    apt-get install -y r-base && \
    rm -rf /var/lib/apt/lists/*

# Install R packages
RUN R -e "install.packages(c('dplyr', 'tidyr', 'yaml', 'caret', 'xgboost'), repos='http://cran.rstudio.com/')"

# Update pip and install dependencies from the requirements.txt file
RUN pip3 install -U pip && \
    pip3 install -r requirements.txt && \
    rm requirements.txt
