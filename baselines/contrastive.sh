#!/bin/sh

if [ -d "models/contrastive" ]
then
    rm -r models/contrastive
fi

mkdir models/contrastive
cp -r models/bert/* models/contrastive

for i in {0..8}
do
   nohup python CONTRASTIVE.py contrastive $i
   nohup python Evaluation.py contrastive $i
done
