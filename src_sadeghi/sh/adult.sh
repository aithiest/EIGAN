pre() {
    python pretrain_fcn.py --expt adult --device gpu --gpu-id 0 1 --optimizer sgd --train-nets gan clf \
	   --n-classes 2 2 --batch-size 256 --test-batch-size 10000 --n-epochs 101 --init-w 1 \
	   --lr-g 0.1 --lr-clfs 0.06 0.006 --alpha 0.5 --weight-decays 1e-6 1e-6
}

adv() {
   python adv_train_fcn.py --expt adult --device gpu --gpu-id 0 1 --optimizer sgd --n-classes 2 2 --batch-size 256 --test-batch-size 10000 --init-w 0 --load-w 1 --ckpt-g ../ckpts/adult/models/pretrain_e.stop --ckpt-clfs ../ckpts/adult/models/pretrain_clf_0.stop ../ckpts/adult/models/pretrain_clf_1.stop --n-epochs 51 --lr-g 0.1 --lr-clfs 0.06 0.006 --ei-array 1.00 -1.00 --weight-decays 1e-6 1e-6
}


check() {
    python check_train_fcn.py --expt adult --device gpu --gpu-id 0 1 --optimizer sgd --n-classes 2 2 --batch-size 4096 --test-batch-size 10000 --init-w 0 --ckpt-g ../ckpts/adult/models/adv_train_lr_g_0.1_lr_clf_0.06_0.006_e.stop --n-epochs 51 --lr-clfs 0.06 0.006 --weight-decays 1e-6
}
$1
