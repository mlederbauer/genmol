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
import pandas as pd
import numpy as np
from tdc import Oracle, Evaluator
from rdkit import DataStructs, Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
from genmol.sampler import Sampler
RDLogger.DisableLog('rdApp.*')


def get_distance(smiles, df):
    if 'MOL' not in df:
        df['MOL'] = df['smiles'].apply(Chem.MolFromSmiles)

    if 'FPS' not in df:
        df['FPS'] = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in df['MOL']]

    fps = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, 1024)
    return np.mean(DataStructs.BulkTanimotoSimilarity(fps, df['FPS'].tolist(), returnDistance=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='hparams.yaml')
    parser.add_argument('-o', '--output', default=None,
                        help='Path to save generated molecules as CSV (e.g. results/frag.csv)')
    args = parser.parse_args()
    config = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), args.config)))

    num_samples = config['num_samples']
    evaluator = Evaluator('diversity')
    oracle_qed = Oracle('qed')
    oracle_sa = Oracle('sa')
    demo = Sampler(config['model_path'])
    data = pd.read_csv('data/fragments.csv')

    tasks = ['linker_design', 'motif_extension', 'scaffold_decoration', 'superstructure_generation', 'linker_design_onestep']
    #! Linker Design / Scaffold Morphing = generate a linker fragment that connects given two side chains
    #! Motif Extension = generate molecule with existing motif
    #! Scaffold Decoration = same as Motif Extension but start with larger scaffold
    #! Superstructure Generation = generate a molecule when a substructure constraint is given
    #! Linker Design (1-step) = generate a linker fragment that connects given two side chains without sequence mixing

    all_results = []

    for task in tasks:
        task_key = task
        if task in ('linker_design', 'scaffold_morphing'):
            task_key = 'linker_design'
            sampling_fn = lambda f: demo.fragment_linking(f, num_samples, **config[task_key])
        elif task in ('motif_extension', 'scaffold_decoration', 'superstructure_generation'):
            sampling_fn = lambda f: demo.fragment_completion(f, num_samples, **config[task_key])
        elif task == 'linker_design_onestep':
            sampling_fn = lambda f: demo.fragment_linking_onestep(f, num_samples, **config[task_key])
            task_key = 'linker_design'

        validity, uniqueness, diversity, distance, quality = [], [], [], [], []
        for name, original, fragment in tqdm(zip(data['name'], data['smiles'], data[task_key]), total=len(data), desc=f'Processing {task}'):
            samples = sampling_fn(fragment)
            if len(samples) == 0:
                validity.append(0)
                uniqueness.append(0)
                quality.append(0)
                continue
            df = pd.DataFrame({'smiles': samples, 'qed': oracle_qed(samples), 'sa': oracle_sa(samples)})
            validity.append(len(df['smiles']) / num_samples)
            df = df.drop_duplicates('smiles')
            uniqueness.append(len(df['smiles']) / len(samples))
            if len(df['smiles']) == 1:
                diversity.append(0)
            else:
                diversity.append(evaluator(df['smiles']))
            distance.append(get_distance(original, df))
            df['quality'] = (df['qed'] >= 0.6) & (df['sa'] <= 4)
            quality.append(df['quality'].sum() / num_samples)

            if args.output:
                df.insert(0, 'fragment', fragment)
                df.insert(0, 'drug_name', name)
                df.insert(0, 'task', task)
                all_results.append(df)

        print(f'{task}')
        print(f'\tValidity:\t{np.mean(validity)}')
        print(f'\tUniqueness:\t{np.mean(uniqueness)}')
        print(f'\tDiversity:\t{np.mean(diversity)}')
        print(f'\tDistance:\t{np.mean(distance)}')
        print(f'\tQuality:\t{np.mean(quality)}')
        print('-' * 50)

    if args.output and all_results:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        out_df = pd.concat(all_results, ignore_index=True)
        out_df.to_csv(args.output, index=False)
        print(f'Saved {len(out_df)} molecules to {args.output}')
