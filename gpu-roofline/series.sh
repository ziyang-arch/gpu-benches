#!/bin/bash

range=1024

mkdir -p build

    
make ./build/$10 N=0 PREFIX=./build 1>&2
make ./build/$11 N=1 PREFIX=./build 1>&2 &
make ./build/$12 N=2 PREFIX=./build 1>&2 &


for (( d=4 ; d<=$range; d+= (d /  24 + 1)*2  ))
do
    echo $d
    make ./build/$1$d N=$d PREFIX=./build 1>&2 &
    while test $(jobs -p | wc -w) -ge 64; do
        sleep 1;
    done
done

while test $(jobs -p | wc -w) -ge 2; do
    echo $(jobs -p | wc -w)
    sleep 1;
done

wait

echo "-- Finished Building --"


./build/$10
./build/$11
./build/$12

if [ -n "$2" ]; then
    output_file="$2"
else
    output_file="Tesla-T4.txt"
fi

> "$output_file"

for (( d=4 ; d<=$range; d+= (d /  24 + 1)*2  ))
do
    ./build/$1$d >> "$output_file"
done
