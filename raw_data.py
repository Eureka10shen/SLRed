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

import numpy as np
import ml_collections
from absl import logging
from typing import Tuple
from chem import (
  VariableStepCVReactor, 
  FixedStepCVReactor, 
  FixedStepCVReactor_dr, 
  stoi_mat
)


def getIDTs(cfg: ml_collections.ConfigDict) -> np.ndarray:
  """Compute IDTs for mechanism reduction conditions.
  """
  idts = []
  # compute idts
  for init_T in cfg.init_Ts:
    for init_P in cfg.init_Ps:
      for phi in cfg.phis:
        idt = VariableStepCVReactor(cfg, init_T, init_P, phi)
        idts.append(idt)
        # logging.debug(f"T={init_T}, P={init_P}, phi={phi}: {idt}.")

  return np.array(idts)


def getRawData(
  cfg: ml_collections.ConfigDict, 
  idts:np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Compute raw data for sparse learning to reduced mechanisms.
  """
  mat = stoi_mat(cfg)
  spe_rates = []
  rct_rates = []
  case_idx = 0
  for init_T in cfg.init_Ts:
    for init_P in cfg.init_Ps:
      for phi in cfg.phis:
        spe_rate, rct_rate = FixedStepCVReactor(
          cfg, init_T, init_P, phi, 2 * idts[case_idx]
        )
        case_idx += 1
        spe_rates.append(spe_rate)
        rct_rates.append(rct_rate)

  return np.array(spe_rates), np.array(rct_rates), np.array(mat)


def getRawData_dr(
  cfg: ml_collections.ConfigDict, 
  idts:np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Compute raw data for detailed reduction method to reduced mechanisms.
  """
  rates_fors = []
  rates_revs = []
  rates_nets = []
  delta_Hs = []
  case_idx = 0
  for init_T in cfg.init_Ts:
    for init_P in cfg.init_Ps:
      for phi in cfg.phis:
        rates_for, rates_rev, rates_net, delta_H = FixedStepCVReactor_dr(
          cfg, init_T, init_P, phi, 2 * idts[case_idx]
        )
        case_idx += 1
        rates_fors.append(rates_for)
        rates_revs.append(rates_rev)
        rates_nets.append(rates_net)
        delta_Hs.append(delta_H)

  return np.array(rates_fors), np.array(rates_revs), np.array(
    rates_nets), np.array(delta_Hs)

