TYPE="gender"
SIZE=80
LR=0.0001
BATCH_SIZE=32
EPOCH=100000
ARCHI="vgg16"
COLOR='rgb'
MEAN="dataset/X_crop_${COLOR}_1_mix_${SIZE}_mean.npy"
X_TRAIN="dataset/X_train_crop_${COLOR}_1_mix_${SIZE}.npy"
T_TRAIN="dataset/T_train_crop_${COLOR}_1_${TYPE}_${SIZE}.npy"
X_TEST="dataset/X_test_crop_${COLOR}_1_mix_${SIZE}.npy"
T_TEST="dataset/T_test_crop_${COLOR}_1_${TYPE}_${SIZE}.npy"

python train.py  --X_train ${X_TRAIN} \
                --T_train ${T_TRAIN} \
                --X_test ${X_TEST} \
                --T_test ${T_TEST} \
                --mean ${MEAN} \
                --logdir ${ARCHI}-${COLOR}-${TYPE}-${SIZE} \
                --lr ${LR} \
                --batch_size ${BATCH_SIZE}  --epoch ${EPOCH} \
                --archi ${ARCHI} \
                --augment 1 \
                ${1} ${2}
