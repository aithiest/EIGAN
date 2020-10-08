mnist () {
    python adv_train.py --expt mnist --device gpu --gpu-id 1 \
	   --optimizer sgd --resnet-layers 34 --n-classes 10 2 2 \
	   --batch-size 256 --init-w 0 --load-w 1\
	   --ckpt-g ../ckpts/mnist/models/pretrain_g.stop \
	   --ckpt-clfs ../ckpts/mnist/models/pretrain_clf_0.stop \
	   ../ckpts/mnist/models/pretrain_clf_1.stop \
	   ../ckpts/mnist/models/pretrain_clf_2.stop \
	   --n-epochs 51 --lr-g 0.001 \
	   --lr-clfs 0.1 0.1 0.1 --ei-array -0.33 0.33 -0.33 --weight-decays 1e-4 1e-4 \
	   --milestones 2 40 70 90 --save-ckpts 10 20 30 40 50 --gamma 0.1
}

$1
