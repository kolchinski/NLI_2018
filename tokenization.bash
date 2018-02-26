# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. 
#

preprocess_exec="sed -f tokenizer.sed"

path=static/snli_1.0

#If you also wish to make the small training set from the original one, then please uncomment the following line
#head -n 10000 $path/snli_1.0_train.txt > $path/snli_1.0_train_small.txt

for split in train train_small dev test
do
    fpath=$path/$split.snli.txt
    awk '{ if ( $1 != "-" ) { print $0; } }' $path/snli_1.0_$split.txt | cut -f 1,6,7 | sed '1d' > $fpath
    cut -f1 $fpath > $path/$split.labels
    cut -f2 $fpath | $preprocess_exec > $path/$split.s1
    cut -f3 $fpath | $preprocess_exec > $path/$split.s2
    rm $fpath
done