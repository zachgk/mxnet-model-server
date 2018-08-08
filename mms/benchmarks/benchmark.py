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
import traceback
from urllib.request import urlretrieve

BENCHMARK_DIR = "/tmp/MMSBenchmark/"

OUT_DIR = os.path.join(BENCHMARK_DIR, 'out/')
RESOURCE_DIR = os.path.join(BENCHMARK_DIR, 'resource/')

RESOURCE_MAP = {
    'kitten.jpg': 'https://s3.amazonaws.com/model-server/inputs/kitten.jpg'
}

JMETER_VERSION = os.listdir('/usr/local/Cellar/jmeter')[0]
CMDRUNNER = '/usr/local/Cellar/jmeter/{}/libexec/lib/ext/CMDRunner.jar'.format(JMETER_VERSION)

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

def run_process(cmd, **kwargs):
    output = None if pargs.verbose else subprocess.DEVNULL
    if pargs.print_commands:
        print(' '.join(cmd) if isinstance(cmd, list) else cmd)
    return subprocess.Popen(cmd, stdout=output, stderr=output, **kwargs)

def run_single_benchmark(jmx, jmeter_args=dict(), threads=10, out_dir=None):
    if out_dir is None:
        out_dir = OUT_DIR
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Build MMS
    if not os.path.exists('../../frontend/server'):
        with ChDir('../../frontend'):
            buildServer_call = './gradlew build'
            buildServer = run_process(buildServer_call, shell=True)
            buildServer.wait()

    # start MMS
    with ChDir('../../frontend/server'):
        startServer_call = '../gradlew startServer'
        startServer = run_process(startServer_call, shell=True)
        startServer.wait()

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
            'threads': threads,
            'loops': pargs.loops,
            'perfmon_file': perfmon_file
        }
        run_jmeter_args.update(jmeter_args)
        abs_jmx = os.path.join(os.getcwd(), 'jmx', jmx)
        jmeter_args_str = ' '.join(['-J{}={}'.format(key, val) for key, val in run_jmeter_args.items()])
        jmeter_call = '/usr/local/bin/jmeter -n -t {} {} -l {} -j {} -e -o {}'.format(abs_jmx, jmeter_args_str, tmpfile, logfile, reportDir)
        jmeter = run_process(jmeter_call.split(' '))
        jmeter.wait()

        # run AggregateReport
        ag_call = 'java -jar {} --tool Reporter --generate-csv {} --input-jtl {} --plugin-type AggregateReport'.format(CMDRUNNER, outfile, tmpfile)
        ag = run_process(ag_call.split(' '))
        ag.wait()

        # Generate output graphs
        gLogfile = os.path.join(out_dir, 'graph_jmeter.log')
        graphing_args = {
            'raw_output': graphsDir,
            'jtl_input': tmpfile
        }
        gabs_jmx = os.path.join(os.getcwd(), 'jmx', 'graphsGenerator.jmx')
        graphing_args_str = ' '.join(['-J{}={}'.format(key, val) for key, val in graphing_args.items()])
        graphing_call = '/usr/local/bin/jmeter -n -t {} {} -j {}'.format(gabs_jmx, graphing_args_str, gLogfile)
        graphing = run_process(graphing_call.split(' '))
        graphing.wait()

        with open(outfile) as f:
            rows = list(csv.DictReader(f))
            inferenceRes = [r for r in rows if r['sampler_label'] == 'Inference Request'][0]
            report = dict(inferenceRes)

    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()

    # kill MMS
    with ChDir('../../frontend/server'):
        killServer_call = '../gradlew killServer'
        killServer = run_process(killServer_call, shell=True)
        killServer.wait()

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
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    reports = dict()
    out_dirs = []
    for i, x in enumerate(xs):
        print("Running value {}={} (value {}/{})".format(key, x, i+1, len(xs)))
        kwargs[key] = x
        out_dir = os.path.join(OUT_DIR, str(i+1))
        out_dirs.append(out_dir)
        report = run_single_benchmark(*args, out_dir=out_dir, **kwargs)
        reports[x] = report

    # files
    merge_results = os.path.join(OUT_DIR, 'merge-results.properties')
    joined = os.path.join(OUT_DIR, 'joined.csv')

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
        merge = run_process(merge_call.split(' '))
        merge.wait()
        shutil.move('joined.csv', joined) # MergeResults ignores path given and puts result into cwd
        baseJtl = joined
        basePrefix = ""

    # plot_multi_latencies(reports, key)
    return reports


class Benchmarks:
    """
    Contains benchmarks to run
    """

    @staticmethod
    def cnn():
        """
        Benchmarks Convolutional Neural Network
        """
        jmeter_args = {'input_filepath': get_resource('kitten.jpg')}
        return run_single_benchmark('resnetPlan.jmx', jmeter_args)

    @staticmethod
    def lstm():
        """
        Benchmarks Residual Neural Network with LSTM
        """
        return run_single_benchmark('lstmPlan.jmx')

    @staticmethod
    def noop():
        """
        Benchmarks without noop model
        """
        return run_single_benchmark('noOpPlan.jmx')

    @staticmethod
    def concurrent_inference():
        """
        Benchmarks number of concurrent inference requests
        """
        jmeter_args = {'input_filepath': get_resource('kitten.jpg')}
        return run_multi_benchmark('threads', range(5, 5*3+1, 5), 'resnetPlan.jmx', jmeter_args)



def run_benchmark(name):
    name = name.lower()
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
        all_benchmarks = [b for b in dir(Benchmarks) if b[0] != '_']
        for benchmark_name in all_benchmarks:
            run_benchmark(benchmark_name)
    else:
        run_benchmark(pargs.name)
