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

from typing import Tuple
import ml_collections
import os
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from absl import logging
logging.set_verbosity(logging.INFO)
from ruamel import yaml as ryaml

import numpy as np
import cantera as ct
from einops import rearrange
from methods import identify_dr
from raw_data import getIDTs, getRawData
from learning import train
from validation import validate
from utils import write


def reduce_gas(gas: ct.Solution, reduced_idx: np.ndarray
) -> Tuple[list, list]:
  """Reduce gas solution using reduced important reaction indices.
  
  Args: 
    gas: Cantera solution of detailed mechanism.
    reduced_idx: Reduced important reaction indices.

  Returns:
    species: Reduced species.
    reactions: Reduced reactions.
  """
  # get reduced species
  spe_idx = set([])
  for i in reduced_idx:
    spe_idx.update(gas.reaction(i).reactants.keys())
    spe_idx.update(gas.reaction(i).products.keys())
  spe_name = list(spe_idx) + ['AR', 'HE', 'N2'] # forced inertial gas
  species = []
  spe_idx = []
  for sp in gas.species():
    if sp.name in spe_name:
      species.append(sp)
      spe_idx.append(gas.species_index(sp.name))

  # get reduced reactions
  reactions = []
  for rct in gas.reactions():
    reac_idx = set(gas.species_index(species) for species in rct.reactants)
    prod_idx = set(gas.species_index(species) for species in rct.products)
    if reac_idx.issubset(set(spe_idx)) and prod_idx.issubset(set(spe_idx)):
      reactions.append(rct)

  return species, reactions


def reduce_mech(cfg: ml_collections.ConfigDict, weight: np.ndarray):
  """Sparse learning reduction using optimized weight.
  """
  errors = np.zeros((1,))
  # set the gas
  gas = ct.Solution(cfg.mechpath)
  gas.TP = 1000.0, 1.0 * ct.one_atm
  fuel = cfg.fuel if type(cfg.fuel) is str else dict(cfg.fuel)
  ox = cfg.ox if type(cfg.ox) is str else dict(cfg.ox)
  gas.set_equivalence_ratio(phi=1.0, fuel=fuel, oxidizer=ox)
  logging.info(f"====== Detailed mechanism ======")
  logging.info(f"Number of species   : {len(gas.species()):5}")
  logging.info(f"Number of reactions : {len(gas.reactions()):5}")

  last_num_spe = gas.n_species
  last_thres = cfg.threshold
  end_thres = 1.0
  while np.max(errors) < cfg.error_limit:
    # indices of important reactions
    reduced_idx = np.where(weight > cfg.threshold)[0]
    species, reactions = reduce_gas(gas, reduced_idx)

    # write into mech file
    red_mech_file = os.path.splitext(cfg.mechpath)[0] + '_' +\
                    f'{len(species):d}' + 'sp.yaml'
    reduced_gas = ct.Solution(
      thermo='ideal-gas', 
      kinetics='gas',
      species=species, 
      reactions=reactions,)    
    logging.info(f"====== Reduced mechanism ======")
    logging.info(f"Number of species   : {len(reduced_gas.species()):5}")
    logging.info(f"Number of reactions : {len(reduced_gas.reactions()):5}")
    write(reduced_gas, cfg.mechpath, red_mech_file)

    # validation
    try:
      errors = validate(cfg, red_mech_file)
    except:
      logging.info("Not ignition under some condition(s).")
      errors = np.ones((1,)) * 100.

    if (np.max(errors) > cfg.error_limit):
      if last_num_spe == gas.n_species: # exceed error limit in first ite
        logging.info("Too high lambda or threshold.")
        break
      elif last_num_spe-1.5 > len(species):
        end_thres = cfg.threshold
        cfg.threshold = (cfg.threshold + last_thres) / 2.
        errors = np.zeros((1,))
    else:
      last_num_spe = len(species)
      last_thres = cfg.threshold
      cfg.threshold = (cfg.threshold + end_thres) / 2.

    if np.abs(last_thres - end_thres) < 1e-6:
      break # prevent overcirculation


def reduced_sl(cfg: ml_collections.ConfigDict):
  """Reduce kinetic mechanism with sparse learning method.
  """
  time0 = time.time()
  dirs = os.path.join('./data', cfg.mech)
  fig_dirs = os.path.join('./figs', cfg.mech)
  if not os.path.exists(dirs):
    os.makedirs(dirs)
  if not os.path.exists(fig_dirs):
    os.makedirs(fig_dirs)

  # generate raw data
  idts_path = os.path.join('./data', cfg.mech, 'idts.npy')
  spe_rates_path = os.path.join('./data', cfg.mech, 'spe_rates.npy')
  rct_rates_path = os.path.join('./data', cfg.mech, 'rct_rates.npy')
  mat_path = os.path.join('./data', cfg.mech, 'mat.npy')
  try:
    idts = np.load(idts_path, allow_pickle=True)
    spe_rates = np.load(spe_rates_path, allow_pickle=True)
    rct_rates = np.load(rct_rates_path, allow_pickle=True)
    mat = np.load(mat_path, allow_pickle=True)
  except:
    idts = getIDTs(cfg)
    np.save(idts_path, idts)
    spe_rates, rct_rates, mat = getRawData(cfg, idts)
    np.save(spe_rates_path, spe_rates)
    np.save(rct_rates_path, rct_rates)
    np.save(mat_path, mat)
  spe_rates = rearrange(spe_rates, 'c t s -> (c t) s')
  rct_rates = rearrange(rct_rates, 'c t r -> (c t) r')
  logging.info(f"Total {idts.shape[0]:3d} 0D cases computed, "
               f"{spe_rates.shape[0]:5d} data shape.")
  time1 = time.time()
  logging.info(f"Generate dataset time : {time1 - time0:10.5f}.")

  # learning
  weight_path = os.path.join('./data', cfg.mech, 'sparse_weight.npy')
  try:
    weight = np.load(weight_path, allow_pickle=True)
  except:
    weight = train(cfg, spe_rates, rct_rates, mat)
  time2 = time.time()
  logging.info(f"Optimize weight time : {time2 - time1:10.5f}.")

  # reduction
  reduce_mech(cfg, weight)
  time3 = time.time()
  logging.info(f"Reduce mechanism time : {time3 - time2:10.5f}.")


def reduce_pymars(input_filename: str):
  """Mechanism reduction using method from PyMARS, e.g., DRGEP, DRGEPSA.
  """
  from pymars.pymars import main, parse_inputs
  # parse input file
  with open(input_filename, 'r') as input_file:
    input_dict = ryaml.YAML(typ='safe', pure=True).load(input_file)
  inputs = parse_inputs(input_dict)

  # reduction
  reduced_model = main(
    inputs.model, 
    inputs.error, 
    inputs.ignition_conditions,
    method                   = inputs.method, 
    target_species           = inputs.target_species,
    safe_species             = inputs.safe_species, 
    phase_name               = inputs.phase_name,
    run_sensitivity_analysis = inputs.sensitivity_analysis, 
    upper_threshold          = inputs.upper_threshold, 
    sensitivity_type         = inputs.sensitivity_type, 
    path                     = "./mechs/",
    num_threads              = 1,)

  # write into mech file
  gas = reduced_model.model
  if inputs.sensitivity_analysis:
    suffix = inputs.method + 'SA'
  else:
    suffix = inputs.method
  red_mech_file = os.path.splitext(inputs.model)[0] + '_' +\
                  f'{gas.n_species:d}' + 'sp_' + suffix + '.yaml'
  write(gas, inputs.model, red_mech_file, 
        description='Reduced mechanism from ' + suffix + ' with pyMARS.')


def reduced_detailed(cfg: ml_collections.ConfigDict):
  """Reduce kinetic mechanism with detailed reduction method.
  """
  errors = np.zeros((1,))
  # set the gas
  gas = ct.Solution(cfg.mechpath)
  last_num_spe = gas.n_species
  last_eps = cfg.eps_r # assume eps_r and eps_q are the same
  end_eps = 1.0
  while np.max(errors) < cfg.error_limit:
    criteria = identify_dr(cfg)
    rct_indices = np.where(criteria)[0]

    # get reduced gas
    spe_idx = set([])
    reactions = []
    for idx in rct_indices:
      reactions.append(gas.reaction(idx))
      spe_idx.update(gas.reaction(idx).reactants.keys())
      spe_idx.update(gas.reaction(idx).products.keys())
    spe_name = list(spe_idx) + ['AR', 'HE', 'N2'] # forced inertial gas
    species = [sp for sp in gas.species() if sp.name in spe_name]

    # get reduced mechanism
    red_mech_file = os.path.splitext(cfg.mechpath)[0] + '_' +\
                    f'{len(species):d}' + 'sp_dr.yaml'
    reduced_gas = ct.Solution(
      thermo='ideal-gas', 
      kinetics='gas',
      species=species, 
      reactions=reactions,)    
    logging.info(f"====== Reduced mechanism ======")
    logging.info(f"Number of species   : {len(reduced_gas.species()):5}")
    logging.info(f"Number of reactions : {len(reduced_gas.reactions()):5}")
    write(reduced_gas, cfg.mechpath, red_mech_file,
          description='Reduced mechanism from detailed reduction method.')

    # validation
    try:
      errors = validate(cfg, red_mech_file)
    except:
      logging.info("Not ignition under some condition(s).")
      errors = np.ones((1,)) * 100.

    # print(errors.max())
    # os.remove(red_mech_file)
    # break

    if (np.max(errors) > cfg.error_limit):
      if last_num_spe == gas.n_species: # exceed error limit in first iteration
        logging.info("Too high initial epsilon.")
        os.remove(red_mech_file)
        break
      elif last_num_spe-1.5 > len(species): # exceed error limit but not fewest species
        end_eps = cfg.eps_r
        cfg.eps_r = (cfg.eps_r + last_eps) / 2.
        cfg.eps_q = (cfg.eps_q + last_eps) / 2. # assume eps_r and eps_q are the same
        errors = np.zeros((1,))
        os.remove(red_mech_file)
    else: # not exceed error limit
      last_path = os.path.splitext(cfg.mechpath)[0] + '_' + f'{last_num_spe:d}' + 'sp_dr.yaml'
      if len(species) < last_num_spe and os.path.exists(last_path):
        os.remove(last_path)
      last_num_spe = len(species)
      last_eps = cfg.eps_r
      cfg.eps_r = (cfg.eps_r + end_eps) / 2.
      cfg.eps_q = (cfg.eps_q + end_eps) / 2.

    if np.abs(last_eps - end_eps) < 1e-6:
      break # prevent overcirculation
  logging.info(f"Final epsilon : eps_r = eps_q = {last_eps:4.2f}.")


