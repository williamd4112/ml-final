MODE="dual"
X="data/child/male"
MODEL_TYPE="train"
CKPT="models/${MODEL_TYPE}-gender.hdf5,models/${MODEL_TYPE}-age.hdf5"
LABEL="mix"
MEAN="dataset/X_crop_rgb_1_mix_80_mean.npy"
SIZE=80
CROP=1
COLOR="rgb"
python test.py --mode ${MODE} --X ${X} --ckpt ${CKPT} --label ${LABEL} --mean ${MEAN} --size ${SIZE} --crop ${CROP} --color ${COLOR}
