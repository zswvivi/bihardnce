#!/bin/sh

if [ -d "models/sent" ]
then
    rm -r models/sent
fi

mkdir models/sent
cp -r models/bert/* models/sent

for i in {0..8}
do
   nohup python SENT_BERT.py sent $i 
   nohup python Evaluation.py sent $i
done
