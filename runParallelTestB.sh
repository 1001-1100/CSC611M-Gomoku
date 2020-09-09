#!/bin/bash

python3 parallelMinMaxB.py 3 3 3 0 3 0

for i in 5 6 7; do
    python3 parallelMinMaxB.py $i $i 5 2 3 1 
done

for i in 5 6 7; do
    python3 parallelMinMaxB.py $i $i 5 2 3 0 
done

for i in 5 6 7; do
    python3 parallelMinMaxB.py $i $i 5 3 3 1 
done