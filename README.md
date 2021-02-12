**NOTE: ACCOMPANYING DATA CAN BE DOWNLOADED FROM THE [DRIVE](https://drive.google.com/file/d/1h1brXcywHgxCEFKzjc0yu6WQMr8IRUvz/view?usp=sharing).**

- **Datasets have to be downloaded individually as per regulations and copyrights.**
- Drive contains the model checkpoints, training histories, and corresponding plots.
- Data is in the same directory structure as required by project (paste in corresponding folders).

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

## Prior works (Baselines)
Checkpoints of prior works baselines:
- [./baselines/BertranAIOI](https://github.com/aithiest/EIGAN/tree/master/baselines/BertranAIOI): [[paper]](http://proceedings.mlr.press/v97/bertran19a.html)
- [./baselines/SadeghiARL](https://github.com/aithiest/EIGAN/tree/master/baselines/SadeghiARL) [[paper]](https://openaccess.thecvf.com/content_CVPRW_2020/html/w1/Sadeghi_Imparting_Fairness_to_Pre-Trained_Biased_Representations_CVPRW_2020_paper.html)


## Training
- all scripts are run from `*.sh` files in `scripts` folder
- change the hyperparameters, as in example scripts
- run the scripts inside the docker container

```shell
sh scipts/<mimic/mnist/titanic>/<script-name>.sh
```

## src folder executions
```shell
cd src
sh sh/<script-name>.sh <expt-name>

# script names: adv_train.sh | check_train.sh | pretrain.sh
# expt-name: mnist
```

## baseline folder executions
```
cd src_sadeghi
sh sh/<script-name>.sh <expt-name>

# script-name: adult.sh | adult_bertran.sh
# expt-name: pre | adv | check
sh sh/adult.sh adv
sh sh/adult.sh check

sh sh/adult_bertran.sh pre
sh sh/adult_bertran.sh adv
sh sh/adult_bertran.sh check
```

- Bertran comparison notebooks in the folder [./baselines/BertranAIOI](https://github.com/aithiest/EIGAN/tree/master/baselines/BertranAIOI)
- Specifically, run the notebooks: [Example-Notebook-Subject-vs-Gender-EIGAN.ipynb](https://github.com/aithiest/EIGAN/blob/master/baselines/BertranAIOI/Example-Notebook-Subject-vs-Gender-EIGAN.ipynb) vs [Example-Notebook-Subject-vs-Gender.ipynb](https://github.com/aithiest/EIGAN/blob/master/baselines/BertranAIOI/Example-Notebook-Subject-vs-Gender.ipynb)

## Comparison
- comparison scripts need editing of python scripts
- replace the names of the pre-populated training histories with the newly generated training histories after training to generate new plots and analysis.

