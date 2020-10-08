mnist () {
    python pretrain.py --expt mnist --device gpu --gpu-id 1 \
	   --train-nets gan clf \
	   --optimizer sgd --resnet-layers 34 --n-classes 10 2 2 \
	   --batch-size 256 --init-w 1 --load-w 0 \
	   --ckpt-g none\
	   --ckpt-d none\
	   --ckpt-clfs none \
	   --n-epochs 51 --lr-g 0.001 --lr-d 0.0001\
	   --lr-clfs 0.1 0.01 0.1 --weight-decays 1e-4 1e-4 1e-4 \
	   --milestones 1 25 40 50 --gamma 0.2
}


$1
