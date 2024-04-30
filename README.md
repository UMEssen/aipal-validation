# AIPAL Validator

AIPAL Validator is a tool designed to streamline the validation process for [AIPAL](https://github.com/VincentAlcazer/AIPAL). Below you'll find instructions on how to set up and run this validator both locally and with Docker.

## Prerequisites

- **R Installation**: Ensure R is installed on your system. If not, install it using:

  ```bash
  sudo apt-get install r-base
  ```

## Local Setup

1. Install the necessary dependencies:

    ```bash
    poetry install
    ```

2. Run the validation process. You can specify the step to run (all, data, sampling, test):

    ```bash
    poetry run aipal_validation --task aipal --step [all,data,sampling,test]
    ```

## Docker Setup

1. Run the Docker container specifying the GPUs (example uses GPUs 0, 1, 2):

    ```bash
    GPUS=0,1,2 docker compose run trainer bash
    ```

2. Inside the Docker container, execute the validation script:

    ```bash
    python -m aipal_validation --task aipal --step [all,data,sampling,test]
    ```
