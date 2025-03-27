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
    'mech': 'Aramco_3_0_propene', # mechanism name
    'mechpath': './mechs/Aramco_3_0_propene_1/Aramco_3_0.yaml', # mechanism filename 
    'fuel': "C3H6", # fuel
    'ox': {"O2": 1.0, "N2": 3.76}, # oxidizer

    # conditions
    'init_Ts': [1000.0, 1200.0, 1400.0, 1600.0], # initial temperature, in [K]
    'init_Ps': [1.0, 5.0, 10.0], # initial pressures, in [atm]
    'phis': [0.5, 1.0, 1.5], # equivalence ratio

    # simulation parameters
    'n_steps': 500, # simulation step for reactors
    'idt_est': 1e0, # estimated maximum IDT, in [s]
    'ign_rise': 400, # ignition temperature rise, in [K]

    # SL parameters
    'lr': 5.e-4, # learning rate
    'batch_size': 4096, # batch size
    'n_epochs': 5000, 
    'lambda_1': 0.5, # balance ratio between two objectives

    # log
    'log_epoch': 100, # log every some epochs

    # reduction parameters
    'threshold': 0.05, # initial threshold for reduced important reactions
    'error_limit': 15, # percent of maximum error
  })

  return cfg



