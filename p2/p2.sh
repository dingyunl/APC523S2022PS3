#!/bin/bash
  
salloc --nodes=1 --ntasks=1 --time=00:01:00

module load anaconda3/2021.11

python3 p2_a.py

python3 p2_b.py

python3 p2_c.py
