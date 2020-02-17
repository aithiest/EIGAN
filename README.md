# EIGAN

## Docker Setup

### To build the docker image
- replace `gpu` with `cpu` in docker-dl-setup/docker-compose.yml in case the system has no gpu
- script to build the docker image
`shell
cd docker-dl-setup
docker-compose build
`

### To run the docker container
`shell
./run-docker.sh
`

### To enter the docker container
`shell
docker exec -it eigan_devel bash
`
