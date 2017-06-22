# Make dataset
```
./make_dataset.sh
```

# Train the model
```
./train_gender.sh
./train_age.sh
```

# Test the model
```
python test.py --mode ${MODE} --X ${X: image data dir} --ckpt ${CKPT} --label ${LABEL: label type} --mean ${MEAN: image mean file path} --size 80 --crop 1 --color rgb
```
