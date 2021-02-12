adult(){
    python pretrain_autoencoder.py --expt adult --device gpu --gpu-id 0 1 2 --net-type linear --optimizer sgd  --encoding-dim 256 --batch-size 16 --test-batch-size 10000 --init-w 1 --load-w 0 --ckpt-g none --n-epochs 51 --lr-g 0.001 --weight-decays 1e-6

    # python pretrain_autoencoder.py --expt adult --device gpu --gpu-id 0 1 2 --net-type fcn --optimizer sgd  --encoding-dim 256 --batch-size 16 --test-batch-size 10000 --init-w 1 --load-w 0 --ckpt-g none --n-epochs 51 --lr-g 0.001 --weight-decays 1e-6 &
}


$1
