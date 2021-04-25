#!/bin/bash

for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./cmake-build-pk-cuda/disarium
    done;