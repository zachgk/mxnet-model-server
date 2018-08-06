# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
ModelServiceWorker is the worker that is started by the MMS front-end.
Communication message format: JSON message
"""

# pylint: disable=redefined-builtin

import argparse
import csv
import matplotlib.pyplot as plt
import os
import pprint
import shutil
import subprocess
import time
from urllib.request import urlretrieve

BENCHMARK_DIR = "/tmp/MMSBenchmark/"
ALL_BENCHMARKS = ['cnn', 'concurrent_inference']

OUT_DIR = os.path.join(BENCHMARK_DIR, 'out/')
RESOURCE_DIR = os.path.join(BENCHMARK_DIR, 'resource/')

RESOURCE_MAP = {
    'cnn.model': 'https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model',
    'kitten.jpg': 'https://s3.amazonaws.com/model-server/inputs/kitten.jpg',
}

def get_resource(name):
    url = RESOURCE_MAP[name]
    path = os.path.join(RESOURCE_DIR, name)
    if not os.path.exists(path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        urlretrieve(url, path)
    return path

def run_process(cmd, **kwargs):
    output = None if pargs.verbose else subprocess.DEVNULL
    if pargs.print_commands:
        print(' '.join(cmd) if isinstance(cmd, list) else cmd)
    return subprocess.Popen(cmd, stdout=output, stderr=output, **kwargs)

def run_single_benchmark(models, jmx, path, jmeter_args=dict(), mms_args='', threads=10):
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    # start MMS
    models_str = ' '.join(['{}={}'.format(name, path) for name, path in models.items()])
    mms_call = 'mxnet-model-server --log-file /dev/null {} --models {}'.format(mms_args, models_str)
    mms = run_process(mms_call, shell=True)
    time.sleep(3)

    # temp files
    tmpfile = os.path.join(OUT_DIR, 'output.jtl')
    logfile = os.path.join(OUT_DIR, 'jmeter.log')
    outfile = os.path.join(OUT_DIR, 'out.csv')

    # run jmeter
    run_jmeter_args = {
        'hostname': 'localhost',
        'port': 8080,
        'threads': threads,
        'loops': pargs.loops,
        'path': path
    }
    run_jmeter_args.update(jmeter_args)
    abs_jmx = os.path.join(os.getcwd(), 'jmx', jmx)
    jmeter_args_str = ' '.join(['-J{}={}'.format(key, val) for key, val in run_jmeter_args.items()])
    jmeter_call = '/usr/local/bin/jmeter -n -t {} {} -l {} -j {}'.format(abs_jmx, jmeter_args_str, tmpfile, logfile)
    jmeter = run_process(jmeter_call.split(' '))
    jmeter.wait()

    # run AggregateReport
    jmeter_version = os.listdir('/usr/local/Cellar/jmeter')[0]
    ag_cmd = '/usr/local/Cellar/jmeter/{}/libexec/lib/ext/CMDRunner.jar'.format(jmeter_version)
    ag_call = 'java -jar {} --tool Reporter --generate-csv {} --input-jtl {} --plugin-type AggregateReport'.format(ag_cmd, outfile, tmpfile)
    ag = run_process(ag_call.split(' '))
    ag.wait()

    mms.kill()

    with open(outfile) as f:
        report = dict(list(csv.DictReader(f))[0])

    return report

def plot_multi_latencies(reports, xlabel):
    keys = sorted(list(reports.keys()))
    line_options = ['average', 'aggregate_report_90%_line', 'aggregate_report_95%_line', 'aggregate_report_99%_line']
    for line in line_options:
        plt.plot(keys, [reports[k][line] for k in keys])
    plt.title('Latencies')
    plt.xlabel(xlabel)
    plt.ylabel('Latency (ms)')
    plt.legend(line_options, loc='upper left')
    plt.savefig(os.path.join(OUT_DIR, 'latencyPercentiles.png'))
    if pargs.output:
        plt.show()
    plt.close()

def run_multi_benchmark(key, xs, *args, **kwargs):
    reports = dict()
    for i, x in enumerate(xs):
        print("Running value {}={} (value {}/{})".format(key, x, i+1, len(xs)))
        kwargs[key] = x
        report = run_single_benchmark(*args, **kwargs)
        reports[x] = report
    plot_multi_latencies(reports, key)
    return reports


class Benchmarks:
    """
    Contains benchmarks to run
    """

    @staticmethod
    def cnn():
        """
        Benchmarks load operation
        """
        models = {'cnn': get_resource('cnn.model')}
        jmeter_args = {'filepath': get_resource('kitten.jpg')}
        return run_single_benchmark(models, 'single.jmx', 'cnn/predict', jmeter_args)

    @staticmethod
    def concurrent_inference():
        """
        Benchmarks number of concurrent inference requests
        """
        models = {'cnn': get_resource('cnn.model')}
        jmeter_args = {'filepath': get_resource('kitten.jpg')}
        return run_multi_benchmark('threads', range(5, 16, 5), models, 'single.jmx', 'cnn/predict', jmeter_args)



def run_benchmark(name):
    if hasattr(Benchmarks, name):
        print("Running benchmark {}".format(name))
        res = getattr(Benchmarks, name)()
        if pargs.output:
            pprint.pprint(res)
        print('\n')
        return(res)
    else:
        raise Exception("No benchmark named {}".format(name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mxnet-model-server-benchmarks', description='Benchmark MXNet Model Server')
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('name', nargs='?', help='The name of the benchmark to run')
    target.add_argument('-a', '--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--output-only', dest='output', action='store_false', help='Don\'t print plots and data')
    parser.add_argument('--print-commands', dest='print_commands', action='store_true', help='Print the commands that are run')
    parser.add_argument('--loops', nargs=1, type=int, default=10, help='Number of loops to run')
    parser.add_argument('-v', '--verbose', action='store_true', help='Display all output')
    pargs = parser.parse_args()

    if pargs.all:
        for benchmark_name in ALL_BENCHMARKS:
            run_benchmark(benchmark_name)
    else:
        run_benchmark(pargs.name)
