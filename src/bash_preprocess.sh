python ../OpenNMT-py/preprocess.py -train_src /data/src_train.csv \
									  -train_tgt /data/tgt_train.csv \
									  -valid_src /data/src_valid.csv \
									  -valid_tgt /data/tgt_valid.csv \
									  -share_vocab \
									  -save_data /data/onmt-data
