# AIPAL Validator

## How to run

```
poetry install
```

```
poetry run aipal_validation --task aipal --step [all,data,sampling,test]
```


## Run with docker

```
GPUS=0,1,2 docker compose run trainer bash
```

and inside the docker container
```
python -m aipal_validation --task aipal --step [all,data,sampling,test]
```
