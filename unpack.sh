#!/bin/bash
# Kevin Patel

echo "unpacking...";
tar -zxvf crunch.tar.gz;
cd crunch;

tar -zxvf raw.tar.gz;
tar -zxvf data.tar.gz;
tar -zxvf mutate.tar.gz;
tar -zxvf recon.tar.gz;
tar -zxvf git.tar.gz;
mv git .git

echo "cleanup...";
rm raw.tar.gz;
rm data.tar.gz;
rm mutate.tar.gz;
rm recon.tar.gz;
rm git.tar.gz;

echo "pull changes:"
git fetch --all;
git reset --hard origin/master;
git pull;
