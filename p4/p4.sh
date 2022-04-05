#!/bin/bash
  
salloc --nodes=1 --ntasks=1 --time=00:30:00

module load anaconda3/2021.11

python3 p4b_RK4.py

python3 p4b_DIRK.py

python3 p4b_BDF2.py
