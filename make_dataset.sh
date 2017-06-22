python dataset.py --mode data --label mix && \
python mix2label.py dataset/T_train_crop_rgb_1_mix_80.npy gender dataset/T_train_crop_rgb_1_gender_80.npy && \
python mix2label.py dataset/T_train_crop_rgb_1_mix_80.npy age dataset/T_train_crop_rgb_1_age_80.npy && \
python mix2label.py dataset/T_test_crop_rgb_1_mix_80.npy gender dataset/T_test_crop_rgb_1_gender_80.npy && \
python mix2label.py dataset/T_test_crop_rgb_1_mix_80.npy age dataset/T_test_crop_rgb_1_age_80.npy
