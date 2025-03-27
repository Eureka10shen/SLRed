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

import ml_collections


def default() -> ml_collections.ConfigDict:
  cfg = ml_collections.ConfigDict({
    # system
    'mech': 'JetSurf', # mechanism name
    'mechpath': './mechs/JetSurf_1_0/JetSurf_1_0.yaml', # mechanism filename 
    'fuel': "NC7H16", # fuel
    'ox': dict({"O2": 1.0, "N2": 3.76}), # oxidizer

    # conditions
    'init_Ts': [1600.0, 1400.0, 1200.0, 1000.0], # initial temperature, in [K]
    'init_Ps': [1.0, 10.0, 30.0], # initial pressures, in [atm]
    'phis': [0.5, 1.0, 1.5], # equivalence ratio

    # simulation parameters
    'n_steps': 2000, # simulation step for reactors
    'idt_est': 1e0, # estimated maximum IDT, in [s]
    'ign_rise': 500, # ignition temperature rise, in [K]

    # SL parameters
    'lr': 5.e-3, # learning rate
    'batch_size': 4096, # batch size
    'n_epochs': 2000, 
    'lambda_1': 0.5, # balance ratio between two objectives

    # log
    'log_epoch': 100, # log every some epochs

    # reduction parameters
    'threshold': 0.1, # initial threshold for reduced important reactions
    'error_limit': 15, # percent of maximum error
  })

  return cfg



