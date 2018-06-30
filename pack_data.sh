#!/bin/bash
# Kevin Patel

echo "packing data...";
tar -czf data.tar.gz data;

mkdir dp;
mv data.tar.gz ./dp/;

echo "packing whole...";
tar -czf dp.tar.gz dp;
mv dp.tar.gz ~;
