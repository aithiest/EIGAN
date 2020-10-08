mnist_resnet50 () {
    python check_train.py --expt mnist --device gpu --gpu-id 1 \
	   --optimizer sgd --resnet-layers 50 --n-classes 10 2 2 \
	   --batch-size 256 --init-w 1 \
	   --ckpt-g ../ckpts/mnist/models/adv_train_g.bkp \
	   --n-epochs 51 \
	   --lr-clfs 0.1 0.1 0.1 --weight-decays 5e-4 \
	   --milestones 2 40 70 90 --gamma 0.1
}

mnist_resnet101 () {
    python check_train.py --expt mnist --device gpu --gpu-id 1 \
	   --optimizer sgd --resnet-layers 101 --n-classes 10 2 2 \
	   --batch-size 256 --init-w 1 \
	   --ckpt-g ../ckpts/mnist/models/adv_train_g.bkp \
	   --n-epochs 51 \
	   --lr-clfs 0.1 0.1 0.1 --weight-decays 5e-4 \
	   --milestones 2 40 70 90 --gamma 0.1
}

mnist_resnet152 () {
    python check_train.py --expt mnist --device gpu --gpu-id 1 \
	   --optimizer sgd --resnet-layers 152  --n-classes 10 2 2 \
	   --batch-size 256 --init-w 1 \
	   --ckpt-g ../ckpts/mnist/models/adv_train_g.bkp \
	   --n-epochs 51 \
	   --lr-clfs 0.1 0.1 0.1 --weight-decays 5e-4 \
	   --milestones 2 40 70 90 --gamma 0.1
}

mnist_resnext101 () {
    python check_train.py --expt mnist --device gpu --gpu-id 1 \
	   --arch resnext --optimizer sgd --num-layers 101  --n-classes 10 2 2 \
	   --batch-size 256 --init-w 1 \
	   --ckpt-g ../ckpts/mnist/models/adv_train_g.bkp \
	   --n-epochs 51 \
	   --lr-clfs 0.1 0.1 0.1 --weight-decays 5e-4 \
	   --milestones 2 40 70 90 --gamma 0.1
}


$1
