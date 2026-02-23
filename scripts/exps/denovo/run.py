# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
sys.path.append(os.path.realpath('.'))

import argparse
import yaml
from time import time
import pandas as pd
from tdc import Oracle, Evaluator
from genmol.sampler import Sampler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='hparams.yaml')
    parser.add_argument('-o', '--output', default=None,
                        help='Path to save generated molecules as CSV (e.g. results/denovo.csv)')
    args = parser.parse_args()
    config = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), args.config)))

    num_samples = config['num_samples']
    evaluator = Evaluator('diversity')
    oracle_qed = Oracle('qed')
    oracle_sa = Oracle('sa')
    sampler = Sampler(config['model_path'])

    t_start = time()
    samples = sampler.de_novo_generation(num_samples,
                                         softmax_temp=config['softmax_temp'],
                                         randomness=config['randomness'],
                                         min_add_len=config['min_add_len'])
    print(f'Time:\t\t{time() - t_start:.2f} sec')
    df = pd.DataFrame({'smiles': samples, 'qed': oracle_qed(samples), 'sa': oracle_sa(samples)})
    print(f'Validity:\t{len(df["smiles"]) / num_samples}')
    df = df.drop_duplicates('smiles')
    print(f'Uniqueness:\t{len(df["smiles"]) / len(samples)}')
    print(f'Diversity:\t{evaluator(df["smiles"])}')
    df['quality'] = (df['qed'] >= 0.6) & (df['sa'] <= 4)
    print(f'Quality:\t{df["quality"].sum() / num_samples}')

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f'Saved {len(df)} molecules to {args.output}')
