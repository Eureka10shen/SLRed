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
import cantera as ct
import ml_collections


def stoi_mat(cfg: ml_collections.ConfigDict) -> np.ndarray:
  """Get stoichiometric matrix of the reaction mechanism.
  """
  mat = []
  # set the gas
  gas = ct.Solution(cfg.mechpath)
  gas.TP = 1000, 1.0 * ct.one_atm
  fuel = cfg.fuel if type(cfg.fuel) is str else dict(cfg.fuel)
  ox = cfg.ox if type(cfg.ox) is str else dict(cfg.ox)
  gas.set_equivalence_ratio(phi=1.0, fuel=fuel, oxidizer=ox)

  for reaction in gas.reactions():
    coeffs = [0] * len(gas.species())
    for reactant, coefficient in reaction.reactants.items():
      coeffs[gas.species_index(reactant)] -= coefficient
    for product, coefficient in reaction.products.items():
      coeffs[gas.species_index(product)] += coefficient
    mat.append(coeffs)
  mat = np.array(mat)

  return mat

