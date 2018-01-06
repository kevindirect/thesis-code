source activate root
python3 split_price.py -f richard@marketpsychdata.com--N164969610.csv
python3 get_trmi.py -v v2
python3 get_trmi.py -v v3
python3 join.py -v v2
python3 join.py -v v3
python3 clean.py -i DJI_v2.csv
python3 clean.py -i DJI_v3.csv
python3 clean.py -i GLD_v2.csv
python3 clean.py -i GLD_v3.csv
python3 clean.py -i NDX_v2.csv
python3 clean.py -i NDX_v3.csv
python3 clean.py -i RUT_v2.csv
python3 clean.py -i RUT_v3.csv
python3 clean.py -i SPX_v2.csv
python3 clean.py -i SPX_v3.csv
python3 clean.py -i USO_v2.csv
python3 clean.py -i USO_v3.csv
source deactivate root
