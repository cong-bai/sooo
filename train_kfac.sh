python train.py --momentum 0 --data-path $HINADORI_LOCAL_SCRATCH --epochs 1 --opt kfac_emp --dataset cifar10 --model timm_vit_base_patch16_224 --lr 0.1 --ema-decay 0.1 --precision fp --accutype fp_s --inverse cholesky
#timm_vit_tiny_patch16_224
