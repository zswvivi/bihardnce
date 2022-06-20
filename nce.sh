#!/bin/sh

if [ -d "models/nce" ]
then
    rm -r models/nce
fi

mkdir models/nce
cp -r models/bert/* models/nce

for i in {0..8}
do
   nohup python NCE.py nce $i 
   nohup python Evaluation.py nce $i
done
