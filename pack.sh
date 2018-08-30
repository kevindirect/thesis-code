#!/bin/bash
# Kevin Patel

echo "packing raw...";
tar -czf raw.tar.gz raw;
echo "packing data...";
tar -czf data.tar.gz data;
echo "packing mutate...";
tar -czf mutate.tar.gz mutate;
echo "packing recon...";
tar -czf recon.tar.gz recon;
echo "packing model...";
tar -czf model.tar.gz model;
echo "packing git...";
tar -czf git.tar.gz .git;

mkdir crunch;
mv raw.tar.gz ./crunch/;
mv data.tar.gz ./crunch/;
mv mutate.tar.gz ./crunch/;
mv recon.tar.gz ./crunch/;
mv model.tar.gz ./crunch/;
mv git.tar.gz ./crunch/;
cp ./common_util.py ./crunch/;
cp ./trmi.json ./crunch/;
cp ./.gitignore ./crunch/;

echo "packing whole...";
tar -czf crunch.tar.gz crunch;
mv crunch.tar.gz ..;
