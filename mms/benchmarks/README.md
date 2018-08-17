# MXNet Model Server Benchmarking

The benchmarks measure the performance of MMS on various models and benchmarks.  It supports either a number of built-in models or a custom model passed in as a path or URL to the .model file.  It also runs various benchmarks using these models (see benchmarks section below).  The benchmarks are run through a python3 script on the user machine through jmeter.  MMS is run on the same machine in a docker instance to avoid network latencies.  The benchmark must be run from within the context of the full MMS repo because it executes the local code as the version of MMS (and it is recompiled between runs) for ease of development.

## Installation

The script is mainly intended to run on a Ubuntu EC2 instance.  For this reason, we have provided an install_dependencies.sh script to install everything needed to execute the benchmark.  All you need to do is run this file and clone the MMS repo.

For other environments, manual installation is necessary.  The list of dependencies to be installed can be found below or by reading the ubuntu installation script.

The benchmarking script requires the following to run:
- python3
- A JDK and JRE
- jmeter installed through homebrew or linuxbrew with the plugin manager and the following plugins:  jpgc-synthesis=2.1,jpgc-filterresults=2.1,jpgc-mergeresults,jpgc-cmd=2.1,jpgc-perfmon=2.1
- Docker-ce with the current user added to the docker group
- Nvidia-docker (for GPU)


## Models

The pre-loaded models for the benchmark can be found in the [MMS model zoo](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md).  We currently support the following:
- [resnet: ResNet-18 (Default)](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md#resnet-18)
- [squeezenet: SqueezeNet V1.1](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md#squeezenet_v1.1)
- [lstm: lstm-ptb](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md#lstm-ptb)
- noop: simple noop model not found in the model zoo (coming soon)

## Benchmarks

We support several basic benchmarks:
- throughput: Run inference with enough threads to occupy all workers and ensure full saturation of resources to find the throughput.  The number of threads is given by the "threads" option and defaults to 100.
- latency: Run inference with a single thread to determine the latency.  The threads can be increased through the "threads" option.
- load: Loads the same model many times in parallel.  The number of loads is given by the "count" option and defaults to 16.
- repeated_scale_calls: Will scale the model up to "scale_up_workers"=16 then down to "scale_down_workers"=1 then up and down repeatedly.

We also support compound benchmarks:
- concurrent_inference: Runs the basic benchmark with different numbers of threads


## Examples

Run basic latency test on default resnet-18 model
`./benchmark.py latency`


Run basic throughput test on default resnet-18 model
`./benchmark.py throughput`


Run all benchmarks
`./benchmark.py --all`


Run using the lstm model
`./benchmark.py latency -m lstm`


Run on GPU (4 gpus)
`./benchmark.py latency -g 4`


Run with a custom image
`./benchmark.py latency -i {imageFilePath}`


Run with a custom model (requires custom image) - currently broken
`./benchmark.py latency -c {modelUrl} -i {imageFilePath}`


Run with custom options
`./benchmark.py repeated_scale_calls --options scale_up_workers 100 scale_down_workers 10`


Run against an already running instance of MMS
`./benchmark.py latency --mms 127.0.0.1:8443
`./benchmark.py latency --mms https://localhost:8443


Run verbose with only a single loop
`./benchmark.py latency -v -l 1`


## Benchmark options

The full list of options can be found by running with the -h or --help flags.


## Frontend Profiling

The benchmarks can be used in conjunction with standard profiling tools such as JProfiler to analyze the system performance.  JProfiler can be downloaded from their [website](https://www.ej-technologies.com/products/jprofiler/overview.html).  Once downloaded, open up JProfiler and follow these steps:

1. Run MMS directly through gradle (do not use docker).  This can be done either on your machine or on a remote machine accessible through SSH.
2. In JProfiler, select "Attach" from the ribbon and attach to the ModelServer.  The process name in the attach window should be "com.amazonaws.ml.mms.ModelServer".  If it is on a remote machine, select "On another computer" in the attach window and enter the SSH details.  For the session startup settings, you can leave it with the defaults.  At this point, you should see live CPU and Memory Usage data on JProfiler's Telemetries section.
3. Select Start Recordings in JProfiler's ribbon
4. Run the Benchmark script targeting your running MMS instance.  It might run something like `./benchmark.py throughput --mms https://localhost:8443`.  It can be run on either your local machine or a remote machine (if you are running remote), but we recommend running the benchmark on the same machine as the model server to avoid confounding network latencies.
5. Once the benchmark script has finished running, select Stop Recordings in JProfiler's ribbon

Once you have stopped recording, you should be able to analyze the data.  One useful section to examine is CPU views > Call Tree and CPU views > Hot Spots to see where the processor time is going.
