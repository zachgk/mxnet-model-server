#!/bin/bash

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# This file contains the installation setup for running benchmarks on EC2 isntance.
# To run on a machine with GPU : ./install_dependencies True
# To run on a machine with CPU : ./install_dependencies False
set -ex

sudo apt-get update
sudo apt-get -y upgrade
echo "Setting up your Ubuntu machine to load test MMS"
sudo apt-get install -y \
        python \
        python-pip \
        default-jre \
        default-jdk \
        linuxbrew-wrapper \
        build-essential

if [[ $1 = True ]]
then
        echo "Installing pip packages for GPU"
        sudo apt install nvidia-cuda-toolkit
        pip install future psutil mxnet-cu92 pillow --user
else
        echo "Installing pip packages for CPU"
        pip install future psutil mxnet pillow --user

fi

echo "Installing JMeter through Brew"
brew install jmeter --with-plugins

echo "Cloning MMS Repository"
git clone https://github.com/awslabs/mxnet-model-server.git --recursive
cd mxnet-model-server/
git checkout netty
git pull origin netty
cd frontend/
./gradlew clean build
cd server/
#killServer startServer
../gradlew kS sS

echo "Checking health of the server"
curl -k -X GET https://localhost:8443/ping

