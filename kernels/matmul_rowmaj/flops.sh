#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 m n k"
    exit 1
fi

m=$1
n=$2
k=$3

echo $((m * n * k * 2))
