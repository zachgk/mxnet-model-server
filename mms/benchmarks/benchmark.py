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
import os
import shutil
import subprocess
import time
from urllib.request import urlretrieve

BENCHMARK_DIR = "/tmp/MMSBenchmark/"
ALL_BENCHMARKS = ['cnn', 'concurrent_inference']

OUT_DIR = os.path.join(BENCHMARK_DIR, 'out/')
MODEL_DIR = os.path.join(BENCHMARK_DIR, 'model/')
CNN = os.path.join(MODEL_DIR, 'cnn.model')
KITTEN = os.path.join(BENCHMARK_DIR, 'kitten')

def get_resource(url, path):
    if not os.path.exists(path):
        directory = os.path.dirname(path)
        if os.path.exists(directory):
            os.makedirs(directory)
        urlretrieve(url, path)

def run_single_benchmark(models, jmx, threads=10):
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    # start MMS
    models_str = ' '.join(['{}={}'.format(name, path) for name, path in models.items()])
    mms_call = 'mxnet-model-server --log-file /dev/null --models {}'.format(models_str)
    mms = subprocess.Popen(mms_call, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    time.sleep(3)

    # temp files
    tmpfile = os.path.join(OUT_DIR, 'output.jtl')
    logfile = os.path.join(OUT_DIR, 'jmeter.log')
    outfile = os.path.join(OUT_DIR, 'out.csv')

    # run jmeter
    jmeter_args = {
        'hostname': 'localhost',
        'port': 8080,
        'threads': threads,
        'loops': 10,
        'filepath': KITTEN
    }
    abs_jmx = os.path.join(os.getcwd(), 'jmx', jmx)
    jmeter_args_str = ' '.join(['-J{}={}'.format(key, val) for key, val in jmeter_args.items()])
    jmeter_call = '/usr/local/bin/jmeter -n -t {} {} -l {} -j {}'.format(abs_jmx, jmeter_args_str, tmpfile, logfile)
    jmeter = subprocess.Popen(jmeter_call.split(' '), stdout=subprocess.PIPE)
    jmeter.wait()
    if args.verbose:
        print(jmeter.stdout.read().decode())

    # run AggregateReport
    jmeter_version = os.listdir('/usr/local/Cellar/jmeter')[0]
    ag_cmd = '/usr/local/Cellar/jmeter/{}/libexec/lib/ext/CMDRunner.jar'.format(jmeter_version)
    ag_call = 'java -jar {} --tool Reporter --generate-csv {} --input-jtl {} --plugin-type AggregateReport'.format(ag_cmd, outfile, tmpfile)
    ag = subprocess.Popen(ag_call.split(' '), stdout=subprocess.PIPE)
    ag.wait()
    if args.verbose:
        print(ag.stdout.read().decode())

    mms.kill()
    if args.verbose:
        print(mms.stdout.read().decode())

    with open(outfile) as f:
        report = dict(list(csv.DictReader(f))[0])

    return report

def run_multi_benchmark(key, xs, *args, **kwargs):
    reports = dict()
    for x in xs:
        kwargs[key] = x
        report = run_single_benchmark(*args, **kwargs)
        reports[x] = report
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
        models = {'cnn': CNN}
        return run_single_benchmark(models, 'basic.jmx')

    @staticmethod
    def concurrent_inference():
        """
        Benchmarks number of concurrent inference requests
        """
        models = {'cnn': CNN}
        return run_multi_benchmark('threads', range(5, 16, 5), models, 'basic.jmx')



def run_benchmark(rname):
    if hasattr(Benchmarks, rname):
        res = getattr(Benchmarks, rname)()
        print(res)
        return(res)
    else:
        raise Exception("No benchmark rnamed {}".format(rname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mxnet-model-server-benchmarks', description='Benchmark MXNet Model Server')
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('name', nargs='?', help='The name of the benchmark to run')
    target.add_argument('-a', '--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('-v', '--verbose', action='store_true', help='Display all output')
    args = parser.parse_args()

    get_resource('https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model', CNN)
    get_resource('https://s3.amazonaws.com/model-server/inputs/kitten.jpg', KITTEN)

    if args.all:
        for name in ALL_BENCHMARKS:
            run_benchmark(name)
    else:
        run_benchmark(args.name)
