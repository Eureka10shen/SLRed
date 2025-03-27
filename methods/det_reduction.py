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

"""Reduce kinetic mechanism with detailed reduction method.
References:
[1] M. Frenklach, K. Kailasanath and E. S. Oran, 
Systematic Development of Reduced Reaction Mechanisms for Dynamic Modeling, 
Progress in Astronautics and Aeronautics 105(2), 365-376 (1986).
[2] H. Wang and M. Frenklach, 
Detailed Reduction of Reaction Mechanisms for Flame Modeling, 
Combust. Flame 87, 365-370 (1991).
[3] M. Frenklach, Reduction of Chemical Reaction Models, 
Numerical Approaches to Combustion Modeling (E. S. Oran and J. P. Boris, Eds.), 
Progress in Astronautics and Aeronautics, Vol. 135, 
American Institute of Aeronautics and Astronautics, Washington, D.C., 1991, pp. 129-154.
[4] Z. Liu, J. Oreluk, A. Hegde, A. Packard, and M. Frenklach, 
Does a reduced model reproduce the uncertainty of the original full-size model? 
Combust. Flame 226, 98-107 (2021).
"""

import ml_collections
import os
from absl import logging
import h5py

import numpy as np
from einops import rearrange
import cantera as ct
from raw_data import getIDTs, getRawData_dr


def identify_dr(cfg: ml_collections.ConfigDict):
  """Identify influential reactions using detailed reduction method."""
  # generate raw data
  raw_path = os.path.join('./data', cfg.mech, 'raw_dr.h5')
  idts_path = os.path.join('./data', cfg.mech, 'idts.npy')
  # get ignition delay times
  try:
    idts = np.load(idts_path, allow_pickle=True)
  except:
    idts = getIDTs(cfg)
    np.save(idts_path, idts)
  # get raw data for detailed reduction
  try:
    with h5py.File(raw_path, "r") as f_raw:
      # forward, reverse and net reaction rates, enthalpy change
      rate_for = np.array(f_raw['rate_for'])
      rate_rev = np.array(f_raw['rate_rev'])
      rate_net = np.array(f_raw['rate_net'])
      delta_H = np.array(f_raw['delta_H'])
  except:
    rate_for, rate_rev, rate_net, delta_H = getRawData_dr(cfg, idts)
    with h5py.File(raw_path, 'w') as f_raw:
      f_raw.create_dataset("rate_for", data=rate_for)
      f_raw.create_dataset("rate_rev", data=rate_rev)
      f_raw.create_dataset("rate_net", data=rate_net)
      f_raw.create_dataset("delta_H", data=delta_H)
  rate_for = rearrange(rate_for, 'c t r -> (c t) r')
  rate_rev = rearrange(rate_rev, 'c t r -> (c t) r')
  rate_net = rearrange(rate_net, 'c t r -> (c t) r')
  delta_H = rearrange(delta_H, 'c t r -> (c t) r')
  logging.info(f"Total {idts.shape[0]:3d} 0D cases computed, "
               f"{rate_for.shape[0] * rate_for.shape[1]:5d} data shape.")

  # detailed reduction
  gas = ct.Solution(cfg.mechpath)
  ref_rct1 = cfg.ref_rct1
  ref_rct2 = cfg.ref_rct2
  ref_idx1, ref_idx2 = -1, -1
  for i in range(gas.n_reactions):
    if gas.reactions()[i].equation == ref_rct1:
      ref_idx1 = i
    if gas.reactions()[i].equation == ref_rct2:
      ref_idx2 = i
    if ref_idx1 != -1 and ref_idx2 != -1:
      break
  if ref_idx1 == -1 or ref_idx2 == -1:
    raise(ref_rct1 + ' or ' + ref_rct2 + ' not found !!!')

  # criteria based on eq(4) from ref[4]
  criteria_1 = np.prod(rate_for < cfg.eps_r * np.maximum(
    rate_net[..., [ref_idx1]], rate_net[..., [ref_idx2]]
  ), axis=0) # shape [reactions]
  criteria_2 = np.prod(rate_rev < cfg.eps_r * np.maximum(
    rate_net[..., [ref_idx1]], rate_net[..., [ref_idx2]]
  ), axis=0)
  criteria_3 = np.prod(np.abs(rate_net * delta_H) < cfg.eps_q * np.max(
    np.abs(rate_net * delta_H), axis=-1, keepdims=True
  ), axis=0)
  criteria = criteria_1 * criteria_2 * criteria_3

  return 1 - criteria

