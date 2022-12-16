#!/bin/zsh

python ./data/custom_datasets.py --path datasets/money/BTC-USD.csv
python ./data/custom_datasets.py --path test/money/BTC-USD-test.csv

python ./BLSTM.py