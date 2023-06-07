export PYTHONPATH=./:$PYTHONPATH
model_dir=/data/user/WZT/models/smoothing/cifar10/resnet110/noise_1.00
mkdir -p $model_dir
python3 train.py \
    --dataset=cifar10 \
    --arch=cifar_resnet110 \
    --outdir=$model_dir \
    --batch=400 \
    --noise=1.00 \
    --gpu=0