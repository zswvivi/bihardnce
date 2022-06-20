#!/bin/sh

if [ -d "models/triplet" ]
then
    rm -r models/triplet
fi

mkdir models/triplet
cp -r models/bert/* models/triplet

for i in {0..8}
do
   nohup python TRIPLET.py triplet $i
   nohup python Evaluation.py triplet $i
done
