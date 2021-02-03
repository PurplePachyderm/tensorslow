#!/bin/bash

# NOTE Run this script from the root of the repository :
# $ bash examples/get-trafficnet.sh

cd examples
mkdir trafficnet
cd trafficnet

wget https://github.com/OlafenwaMoses/Traffic-Net/releases/download/1.0/trafficnet_dataset_v1.zip

unzip trafficnet_dataset_v1.zip
rm trafficnet_dataset_v1.zip

mv trafficnet_dataset_v1/test test
mv trafficnet_dataset_v1/train train
rm -rf trafficnet_dataset_v1
mkdir results
