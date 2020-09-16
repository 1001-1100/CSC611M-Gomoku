#!/bin/bash

for i in 5 6 7; do
    python3 parallelMinMax.py $i $i 5 5 3 0 
done

for i in 5 6 7; do
    python3 parallelMinMax.py $i $i 5 5 3 1 
done