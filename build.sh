#!/usr/bin/bash

echo "start compiling..."
compile1=`g++ -larmadillo -O3 -std=c++14 -fPIC -larmadillo -lmlpack -fopenmp main.cpp -o tmp/coa_parallel 2>/dev/null`;
compile2=`g++ -larmadillo -O3 -std=c++14 -fPIC -larmadillo -lmlpack          main.cpp -o tmp/coa_serial   2>/dev/null`;
echo "All files compiled successfully!";

serial=`tmp/coa_serial`;
serial_time=`echo $serial / 1000000| bc -l`
printf "serial   time is %.2f sec\n" $serial_time;

parallel=`tmp/coa_parallel`;
parallel_time=`echo $parallel / 1000000 | bc -l`
printf "parallel time is %.2f sec\n" $parallel_time;

speedup=`echo $serial_time / $parallel_time | bc -l`;
printf "speedup =        %.2f\n" $speedup;
