# Copyright 2024 Shen Fang, Beihang University. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from absl import logging
logging.set_verbosity(logging.INFO)
import argparse
import warnings
import importlib
from reduction import reduced_detailed, reduced_sl, reduce_pymars


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Kinetic mechanism reduction')
  parser.add_argument('--fuel', type=str, default='n_heptane', help='Name of fuel')
  parser.add_argument('--method', type=str, default='sl', help='Reduction method')
  args = parser.parse_args()

  if args.method == 'sl': # sparse learning method
    try:
      module_name = f'cfgs.{args.fuel}'
      module = importlib.import_module(module_name)
      default = module.default
      print(f"Successfully imported 'default' from {module_name}")
    except ImportError as e:
      print(f"Error: Unable to import 'default' from {module_name}. {e}")
    # sparse learning reduction
    cfg = default()
    reduced_sl(cfg)
  elif args.method == 'dr': # detailed reduction method
    try:
      module_name = f'cfgs.{args.fuel}'
      module = importlib.import_module(module_name)
      default = module.default
      print(f"Successfully imported 'default' from {module_name}")
    except ImportError as e:
      print(f"Error: Unable to import 'default' from {module_name}. {e}")
    # detailed reduction
    cfg = default()
    cfg.unlock()
    cfg.eps_r = 0.1 # initial epsilon
    cfg.eps_q = 0.1
    cfg.ref_rct1 = 'H + O2 <=> O + OH' # user-defined reference reactions
    cfg.ref_rct2 = 'H + O2 (+M) <=> HO2 (+M)'
    cfg.n_steps = 100
    cfg.error_limit = 15
    cfg.lock()
    reduced_detailed(cfg)
  else: # methods from pyMARS, e.g. DRGEP and DRGEPSA
    cfg_file = os.path.join('./cfgs', args.fuel+'.yaml')
    reduce_pymars(cfg_file)

