#!/bin/bash

#PBS -q titanx

cd $SCRATCH

execdir=/home/ParallelImageQuilting/parallel
exe=ImageQuilt

# change this for image
args="rice.jpg b"

cp ${execdir}/${exe} ${exe}
cp ${execdir}/*.jpg .

./${exe} ${args}

cp *.jpg ${execdir}

