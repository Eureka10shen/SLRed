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
from typing import Union


def OneDimFlame(
  gas: ct.Solution, fuel: Union[str, dict], ox: Union[str, dict], 
  init_T: float, init_P: float, phi: Union[float, dict], 
  width: float=0.03, # ratio: float=10, slope: float=0.05, curve: float=0.1,
  ratio: float=3, slope: float=0.06, curve: float=0.10,
) -> float:
  """One-dimensional free flame to generate laminar flame speeds.

  Args:
    gas: Gas solution.
    init_T: Initial temperature of 0D reactor.
    init_P: Initial pressure of 0D reactor.
    phi: Equivalence ratio (float) or mass fraction (list) of 0D reactor.

  Returns:
    idt: Laminar flame speed, in [m/s].
  """
  # set the gas
  gas.TP = init_T, init_P * ct.one_atm
  if type(phi) is float or type(phi) is np.float64:
    gas.set_equivalence_ratio(phi=phi, fuel=fuel, oxidizer=ox)
  else:
    gas.Y = phi

  # set the flame
  f = ct.FreeFlame(gas, width=width)
  f.set_refine_criteria(ratio=ratio, slope=slope, curve=curve)
  f.transport_model = 'mixture-averaged'

  # compute the LFS
  f.solve(loglevel=0, auto=False)

  return f.velocity[0]


