#!/bin/bash

for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./cmake-build-pk-cuda/bradley /home/students/2021DS/grkrol/prir/test_photos/city2.jpg
    done;