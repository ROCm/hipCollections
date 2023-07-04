# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import subprocess
import re
import glob
import itertools
from itertools import chain
import sys
import argparse
import pandas as pd

# from colorama import Fore, Back, Style
# from tqdm import tqdm


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, new):
        self.new = os.path.expanduser(new)

    def __enter__(self):
        self.saved = os.getcwd()
        os.chdir(self.new)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved)

def runCommand(cmd):
    proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    if err:
        print(f"There was an error running command {cmd}!")
    return str(out)

def extract_substring(text, start_token, end_token):
    # start_token = "| "
    # end_token = " ms"
    start_index = text.find(start_token)
    if start_index == -1:
        return None
    start_index += len(start_token)
    end_index = text.find(end_token, start_index)
    if end_index == -1:
        return None
    substring = text[start_index:end_index]
    return substring

import csv

def calculate_average(csv_file, metrics):
    # Initialize variables
    total_rows = 0

    sum_values = {m: 0 for m in metrics}

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)

        # Process each row
        for row in reader:
            total_rows += 1

            # Accumulate column values for average calculation
            for column in sum_values:
                sum_values[column] += float(row[column])

    # Calculate average values
    average_values = {column: sum_values[column] / total_rows for column in sum_values}

    return average_values
    # return sum_values

metrics = [  # Columns to calculate average
        'FETCH_SIZE',
        'WRITE_SIZE',
        'TCC_EA_RDREQ_sum',
        'TCC_EA_WRREQ_sum',
        'TCP_TCC_READ_REQ_sum',
        'TCP_TCC_WRITE_REQ_sum',
        'TCP_TCC_READ_REQ_LATENCY_sum',
        'TCP_TCC_WRITE_REQ_LATENCY_sum',
        'TCC_EA_WRREQ_DRAM_sum',
        'TCC_EA_RDREQ_DRAM_sum',
        'TCC_READ_sum',
        'TCC_WRITE_sum',
        'TCC_HIT_sum',
        'TCC_MISS_sum',
        'TCC_NORMAL_EVICT_sum',
        'TCC_ALL_TC_OP_INV_EVICT_sum'
    ]

output_file = 'averages.csv'
csv_file_path = 'counters_amd.csv' 

averages = {}
input = [100000000, 10000000, 8000000, 5242880, 5000000, 1000000, 500000]

output_file = 'output.xlsx'
df = pd.DataFrame(columns=['Problem_size','Time (s)','TCC_HIT_rate'] + metrics)

# 100000000, 10000000, 8000000, 5242880, 5000000, 1000000, 500000
# with open(output_file, 'w', newline='') as file:
#     writer = csv.writer(file)

for i, value in enumerate(input):
    cloneCmd = "rocprof --basenames on --timestamp on -i /home/projects/7/hipCollections/benchmarks/hash_table/static_multimap_probing_bench/counters_amd.txt /home/projects/7/hipCollections/build/benchmarks/STATIC_MULTIMAP_PROBING_BENCH --devices 0 -a Multiplicity=1  -a Key=I32 -a Value=I32 -a INPUTSize={0}".format(value)
    out = runCommand(cloneCmd)
    # print(out)
    elapsed_time = extract_substring(out, "% | ","s | ")
    print(elapsed_time)
    averages['Problem_size'] = value
    averages['Time (s)'] = elapsed_time
    averages.update(calculate_average(csv_file_path, metrics))  
    # writer.writerow([averages])
    averages['TCC_HIT_rate'] = (averages['TCC_HIT_sum'] / (averages['TCC_READ_sum'] + averages['TCC_WRITE_sum']))*100
    df.loc[i] = averages

if os.path.exists(output_file):
    os.remove(output_file)

df.to_excel(output_file, index=False)
