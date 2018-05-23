#!/bin/bash
wget 'https://www.dropbox.com/s/r5zb4apxlia7eo5/w2vector'
wget 'https://www.dropbox.com/s/y5exoqo678ed38v/w2vector.syn1neg.npy'
wget 'https://www.dropbox.com/s/h49rmux2ltqyb7i/w2vector.wv.syn0.npy'
python3 hw5_test.py $1 $2