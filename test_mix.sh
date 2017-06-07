LABEL="mix"
SIZE=80
ARCHI="vgg16"
COLOR="rgb"
CROP=1
MODE="dual"
MEAN="dataset/X_crop_${COLOR}_1_mix_${SIZE}_mean.npy"

X="demo"
CKPT=$1

python test.py  --X ${X} \
                --mean ${MEAN} \
                --mode ${MODE} \
                --label ${LABEL} \
                --size ${SIZE} \
                --crop ${CROP} \
                --color ${COLOR} \
                --ckpt ${CKPT}
            

