#!/usr/bin/env python3

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
Execute the MMS Benchmark.  For instructions, run with the --help flag
"""

# pylint: disable=redefined-builtin

import argparse
import csv
import itertools
import os
import pprint
import shutil
import subprocess
import sys
import time
import traceback
from functools import reduce
from urllib.request import urlretrieve

BENCHMARK_DIR = "/tmp/MMSBenchmark/"

OUT_DIR = os.path.join(BENCHMARK_DIR, 'out/')
RESOURCE_DIR = os.path.join(BENCHMARK_DIR, 'resource/')

RESOURCE_MAP = {
    'kitten.jpg': 'https://s3.amazonaws.com/model-server/inputs/kitten.jpg'
}

MODEL_MAP = {
    'squeezenet': ('imageInputModelPlan.jmx', {'url': 'https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model', 'model_name': 'squeezenet', 'input_filepath': 'kitten.jpg'}),
    'resnet': ('imageInputModelPlan.jmx', {'url': 'https://s3.amazonaws.com/model-server/models/resnet-18/resnet-18.model', 'model_name': 'resnet-18', 'input_filepath': 'kitten.jpg'}),
    'lstm': ('textInputModelPlan.jmx', {'url': 'https://s3.amazonaws.com/model-server/models/lstm_ptb/lstm_ptb.model'}),
    'noop': ('textInputModelPlan.jmx')
}

JMETER_RESULT_SETTINGS = {
    'jmeter.reportgenerator.overall_granularity': 1000,
    # 'jmeter.reportgenerator.report_title': '"MMS Benchmark Report Dashboard"',
    'aggregate_rpt_pct1': 50,
    'aggregate_rpt_pct2': 90,
    'aggregate_rpt_pct3': 99,
}

CELLAR = '/home/ubuntu/.linuxbrew/Cellar/jmeter' if 'linux' in sys.platform else '/usr/local/Cellar/jmeter'
JMETER_VERSION = os.listdir(CELLAR)[0]
CMDRUNNER = '{}/{}/libexec/lib/ext/CMDRunner.jar'.format(CELLAR, JMETER_VERSION)
JMETER = '{}/{}/libexec/bin/jmeter'.format(CELLAR, JMETER_VERSION)
MMS_BASE = reduce(lambda val,func: func(val), (os.path.abspath(__file__),) + (os.path.dirname,) * 3)


ALL_BENCHMARKS = itertools.product(('single', 'concurrent_inference'), ('resnet', 'lstm', 'noop'))

class ChDir:
    def __init__(self, path):
        self.curPath = os.getcwd()
        self.path = path

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, *args):
        os.chdir(self.curPath)


def get_resource(name):
    url = RESOURCE_MAP[name]
    path = os.path.join(RESOURCE_DIR, name)
    if not os.path.exists(path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        urlretrieve(url, path)
    return path

def run_process(cmd, wait=True, **kwargs):
    output = None if pargs.verbose else subprocess.DEVNULL
    if pargs.verbose:
        print(' '.join(cmd) if isinstance(cmd, list) else cmd)
    if not kwargs.get('shell') and isinstance(cmd, str):
        cmd = cmd.split(' ')
    if 'stdout' not in kwargs:
        kwargs['stdout'] = output
    if 'stderr' not in kwargs:
        kwargs['stderr'] = output
    p = subprocess.Popen(cmd, **kwargs)
    if wait:
        p.wait()
    return p

def run_single_benchmark(jmx, jmeter_args=dict(), threads=10, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(OUT_DIR, benchmark_name, pargs.model)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Start MMS
    docker = 'nvidia-docker' if pargs.gpus else 'docker'
    container = 'mms_benchmark_gpu' if pargs.gpus else 'mms_benchmark_cpu'
    docker_check_call = "{} ps".format(docker)
    docker_check = run_process(docker_check_call, stdout=subprocess.PIPE)
    docker_check_all_call = "{} ps -a".format(docker)
    docker_check_all = run_process(docker_check_all_call, stdout=subprocess.PIPE)
    if container not in docker_check.stdout.read().decode():
        if container in docker_check_all.stdout.read().decode():
            docker_run_call = "{} start {}".format(docker, container)
            run_process(docker_run_call)
        else:
            docker_run_call = "{} run --name {} -p 8080:8080 -v {}:/mxnet-model-server -itd vamshidhardk/mms_netty".format(docker, container, MMS_BASE)
            run_process(docker_run_call)
            run_process("{} exec -it {} sh -c 'cd /mxnet-model-server && pip install -U -e .'".format(docker, container), shell=True)


    with ChDir(MMS_BASE):
        run_process('python setup.py bdist_wheel --universal')
    docker_start_call = "{} exec {} mxnet-model-server --start".format(docker, container)
    docker_start = run_process(docker_start_call, wait=False)
    time.sleep(3)
    docker_start.kill()



    try:
        # temp files
        tmpfile = os.path.join(out_dir, 'output.jtl')
        logfile = os.path.join(out_dir, 'jmeter.log')
        outfile = os.path.join(out_dir, 'out.csv')
        perfmon_file = os.path.join(out_dir, 'perfmon.csv')
        graphsDir = os.path.join(out_dir, 'graphs')
        reportDir = os.path.join(out_dir, 'report')

        # run jmeter
        run_jmeter_args = {
            'hostname': '127.0.0.1',
            'port': 8080,
            'protocol': 'http',
            'min_workers': 2,
            'rampup': 5,
            'threads': threads,
            'loops': pargs.loops,
            'perfmon_file': perfmon_file
        }
        if pargs.gpus:
            run_jmeter_args['number_gpu'] = '&numberGpu={}'.format(pargs.gpus)
        run_jmeter_args.update(JMETER_RESULT_SETTINGS)
        run_jmeter_args.update(jmeter_args)
        abs_jmx = os.path.join(os.getcwd(), 'jmx', jmx)
        jmeter_args_str = ' '.join(['-J{}={}'.format(key, val) for key, val in run_jmeter_args.items()])
        jmeter_call = '{} -n -t {} {} -l {} -j {} -e -o {}'.format(JMETER, abs_jmx, jmeter_args_str, tmpfile, logfile, reportDir)
        run_process(jmeter_call)

        # run AggregateReport
        ag_call = 'java -jar {} --tool Reporter --generate-csv {} --input-jtl {} --plugin-type AggregateReport'.format(CMDRUNNER, outfile, tmpfile)
        run_process(ag_call)

        # Generate output graphs
        gLogfile = os.path.join(out_dir, 'graph_jmeter.log')
        graphing_args = {
            'raw_output': graphsDir,
            'jtl_input': tmpfile
        }
        graphing_args.update(JMETER_RESULT_SETTINGS)
        gabs_jmx = os.path.join(os.getcwd(), 'jmx', 'graphsGenerator.jmx')
        graphing_args_str = ' '.join(['-J{}={}'.format(key, val) for key, val in graphing_args.items()])
        graphing_call = '{} -n -t {} {} -j {}'.format(JMETER, gabs_jmx, graphing_args_str, gLogfile)
        run_process(graphing_call)

        print("Output available at {}".format(out_dir))
        print("Report generated at {}".format(os.path.join(reportDir, 'index.html')))

        with open(outfile) as f:
            rows = list(csv.DictReader(f))
            inferenceRes = [r for r in rows if r['sampler_label'] == 'Inference Request'][0]
            report = dict(inferenceRes)

        return report

    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()

def run_multi_benchmark(key, xs, *args, **kwargs):
    out_dir = os.path.join(OUT_DIR, benchmark_name, pargs.model)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    reports = dict()
    out_dirs = []
    for i, x in enumerate(xs):
        print("Running value {}={} (value {}/{})".format(key, x, i+1, len(xs)))
        kwargs[key] = x
        sub_out_dir = os.path.join(out_dir, str(i+1))
        out_dirs.append(sub_out_dir)
        report = run_single_benchmark(*args, out_dir=sub_out_dir, **kwargs)
        reports[x] = report

    # files
    merge_results = os.path.join(out_dir, 'merge-results.properties')
    joined = os.path.join(out_dir, 'joined.csv')

    # merge runs together
    inputJtls = [os.path.join(out_dirs[i], 'output.jtl') for i in range(len(xs))]
    prefixes = ["{} {}: ".format(key, x) for x in xs]
    baseJtl = inputJtls[0]
    basePrefix = prefixes[0]
    for i in range(1, len(xs), 3): # MergeResults only joins up to 4 at a time
        with open(merge_results, 'w') as f:
            curInputJtls = [baseJtl] + inputJtls[i:i+3]
            curPrefixes = [basePrefix] + prefixes[i:i+3]
            for j, (jtl, p) in enumerate(zip(curInputJtls, curPrefixes)):
                f.write("inputJtl{}={}\n".format(j+1, jtl))
                f.write("prefixLabel{}={}\n".format(j+1, p))
                f.write("\n")
        merge_call = 'java -jar {} --tool Reporter --generate-csv joined.csv --input-jtl {} --plugin-type MergeResults'.format(CMDRUNNER, merge_results)
        run_process(merge_call)
        shutil.move('joined.csv', joined) # MergeResults ignores path given and puts result into cwd
        baseJtl = joined
        basePrefix = ""

    return reports

def parseModel():
    if benchmark_model in MODEL_MAP:
        plan, jmeter_args = MODEL_MAP[benchmark_model]
        for k, v in jmeter_args.items():
            if v in RESOURCE_MAP:
                jmeter_args[k] = get_resource(v)
        return plan, jmeter_args
    else:
        raise Exception('Unknown model')

class Benchmarks:
    """
    Contains benchmarks to run
    """

    @staticmethod
    def single():
        """
        Performs simple single benchmark
        """
        plan, jmeter_args = parseModel()
        return run_single_benchmark(plan, jmeter_args)

    @staticmethod
    def concurrent_inference():
        """
        Benchmarks number of concurrent inference requests
        """
        plan, jmeter_args = parseModel()
        return run_multi_benchmark('threads', range(5, 5*3+1, 5), plan, jmeter_args)



def run_benchmark():
    if hasattr(Benchmarks, benchmark_name):
        print("Running benchmark {} with model {}".format(benchmark_name, benchmark_model))
        res = getattr(Benchmarks, benchmark_name)()
        if pargs.output:
            pprint.pprint(res)
        print('\n')
        return(res)
    else:
        raise Exception("No benchmark benchmark_named {}".format(benchmark_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mxnet-model-server-benchmarks', description='Benchmark MXNet Model Server')

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('name', nargs='?', help='The name of the benchmark to run')
    target.add_argument('-a', '--all', action='store_true', help='Run all benchmarks')

    parser.add_argument('-m', '--model', nargs=1, type=str, dest='model', default='resnet', help='The model to run.  Can be a url or one of the defaults (squeezenet, lstm, resnet, noop)')
    parser.add_argument('-g', '--gpus', nargs=1, type=int, default=0, help='Number of gpus.  Specify to enable GPU call')
    parser.add_argument('--output-only', dest='output', action='store_false', help='Don\'t print plots and data')
    parser.add_argument('--loops', nargs=1, type=int, default=10, help='Number of loops to run')
    parser.add_argument('-v', '--verbose', action='store_true', help='Display all output')
    pargs = parser.parse_args()

    if pargs.all:
        for benchmark_name, benchmark_model in ALL_BENCHMARKS:
            run_benchmark()
    else:
        benchmark_name = pargs.name.lower()
        benchmark_model = pargs.model.lower()
        run_benchmark()
