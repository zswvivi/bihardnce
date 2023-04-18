#!/bin/sh

# Check if the "models/bihardnce" directory exists
# If the directory exists, delete it

if [ -d "models/bihardnce" ] 
then
    rm -r models/bihardnce 
fi

# Create a new directory named "bihardnce" inside the "models" directory
mkdir models/bihardnce

# Copy Chinese BERT from "models/bert" to "models/bihardnce"
cp -r models/bert/* models/bihardnce


# Loop through values (epoch) of i from 0 to 8 
for i in {0..8}
do
   nohup python BiHDNCE.py bihardnce $i 
   nohup python Evaluation.py bihardnce $i
done
