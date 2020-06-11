**NOTE: ACCOMPANYING DATA (MODELS AND CHECKPOINTS) CAN BE DOWNLOADED FROM THE [DRIVE](https://drive.google.com/drive/folders/1K8j1J5B6SNB3zmnmHgVidQnbLGUUTFbo?usp=sharing)**

**Datasets have to be downloaded individually as per regulations and copyrights.**

# EIGAN

- docker integration is used to reduce the overhead of setting up environment.
- users are welcome to use non-docker environments on their own.
- prepopulated hyperparameters and training logs as well as pretrained models are made available for evaluation.


## Docker Setup

### To build the docker image
- replace `gpu` with `cpu` in `docker-dl-setup/docker-compose.yml` in case the system has no gpu
- script to build the docker image

```shell
cd docker-dl-setup
docker-compose build
```

### To run the docker container

```shell
./run-docker.sh
```

### To enter the docker container

```shell
docker exec -it eigan_devel bash
```

## Training
- all scripts are run from `*.sh` files in `scripts` folder
- change the hyperparameters, as in example scripts
- run the scripts inside the docker container

```shell
sh scipts/<mimic/mnist/titanic>/<script-name>.sh
```

## Comparison
- comparison scripts need editing of python scripts
- replace the names of the pre-populated training histories with the newly generated training histories after training to generate new plots and analysis.

