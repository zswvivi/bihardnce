#!/bin/sh

if [ -d "models/bihardnce" ] 
then
    rm -r models/bihardnce 
fi

mkdir models/bihardnce
cp -r models/bert/* models/bihardnce

for i in {0..8}
do
   nohup python BiHDNCE.py bihardnce $i 
   nohup python Evaluation.py bihardnce $i
done
