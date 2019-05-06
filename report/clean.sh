echo 'Remember to call this script from the project root directory'
echo 'deleting logfiles'
rm logfile.txt
rm hmw_logfile.txt
echo 'deleting db'
rm -rf ./report/db/
echo 'deleting dow_jones'
rm -rf ./report/dow_jones/
echo 'deleting sp_500'
rm -rf ./report/sp_500/
echo 'deleting nasdaq_100'
rm -rf ./report/nasdaq_100/
echo 'deleting russell_2000'
rm -rf ./report/russell_2000/
