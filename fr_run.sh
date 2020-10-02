#!/bin/sh
# Run feature rank script with multiple instances

python3 -m model.fr -w 1 -k 3 &
python3 -m model.fr -w 3 -k 3 &
python3 -m model.fr -w 5 -k 3 &
python3 -m model.fr -w 10 -k 3 &
python3 -m model.fr -w 20 -k 3;

python3 -m model.fr -w 1 -k 7 &
python3 -m model.fr -w 3 -k 7 &
python3 -m model.fr -w 5 -k 7 &
python3 -m model.fr -w 10 -k 7 &
python3 -m model.fr -w 20 -k 7;
