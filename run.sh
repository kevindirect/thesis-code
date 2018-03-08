# Kevin Patel
source activate root
python3 -m raw.get_trmi -k
python3 -m raw.get_price
python3 -m raw.dump_raw
source deactivate root