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
from typing import Tuple, Union


def FixedStepCVReactor(
  cfg: ml_collections.ConfigDict, 
  init_T: float, init_P: float, phi: Union[float, dict], 
  time: float, 
) -> Tuple[np.ndarray, np.ndarray]:
  """Fixed step constant volume reactor to generate dataset for learning reduced
  chemical kinetic mechanisms.

  Args:
    cfg: Config dict that stores mechanism and condition information.
    init_T: Initial temperature of 0D reactor.
    init_P: Initial pressure of 0D reactor.
    phi: Equivalence ratio (float) or mass fraction (list) of 0D reactor.
    time: Simulation time, default 2 \times IDT.

  Returns:
    spe_rate: Net production rates of species, in [kmol/m^3/s].
    rct_rate: Net progress rate of reactions, in [kmol/m^3/s].
  """
  spe_rate = []
  rct_rate = []
  # set the gas
  gas = ct.Solution(cfg.mechpath)
  gas.TP = init_T, init_P * ct.one_atm
  fuel = cfg.fuel if type(cfg.fuel) is str else dict(cfg.fuel)
  ox = cfg.ox if type(cfg.ox) is str else dict(cfg.ox)
  if type(phi) is float:
    gas.set_equivalence_ratio(phi=phi, fuel=fuel, oxidizer=ox)
  else:
    gas.Y = phi

  # set the reactor
  r = ct.IdealGasMoleReactor(contents=gas)
  sim = ct.ReactorNet([r])
  sim.verbose = False

  # compute reaction progress
  dt = time / cfg.n_steps
  for _ in range(cfg.n_steps):
    sim.advance(sim.time + dt)
    spe_rate.append(r.kinetics.net_production_rates)
    rct_rate.append(r.kinetics.net_rates_of_progress)

  return np.array(spe_rate), np.array(rct_rate)


def FixedStepCVReactor_dr(
  cfg: ml_collections.ConfigDict, 
  init_T: float, init_P: float, phi: Union[float, dict], 
  time: float, 
) -> Tuple[np.ndarray, np.ndarray]:
  """Fixed step constant volume reactor to generate dataset for learning reduced
  chemical kinetic mechanisms using detailed reduction method.

  Args:
    cfg: Config dict that stores mechanism and condition information.
    init_T: Initial temperature of 0D reactor.
    init_P: Initial pressure of 0D reactor.
    phi: Equivalence ratio (float) or mass fraction (list) of 0D reactor.
    time: Simulation time, default 2 \times IDT.

  Returns:
    rate_for: Forward rates of progress for the reactions, in [kmol/m^3/s].
    rate_rev: Reverse rates of progress for the reactions, in [kmol/m^3/s].
    rate_net: Net rates of progress for the reactions, in [kmol/m^3/s].
    delta_H: Enthalpy change in the rection, in [J/kmol].
  """
  rate_for, rate_rev, rate_net, delta_H = [], [], [], []
  # set the gas
  gas = ct.Solution(cfg.mechpath)
  gas.TP = init_T, init_P * ct.one_atm
  fuel = cfg.fuel if type(cfg.fuel) is str else dict(cfg.fuel)
  ox = cfg.ox if type(cfg.ox) is str else dict(cfg.ox)
  if type(phi) is float:
    gas.set_equivalence_ratio(phi=phi, fuel=fuel, oxidizer=ox)
  else:
    gas.Y = phi

  # set the reactor
  r = ct.IdealGasMoleReactor(contents=gas)
  sim = ct.ReactorNet([r])
  sim.verbose = False

  # compute reaction progress
  dt = time / cfg.n_steps
  for _ in range(cfg.n_steps):
    sim.advance(sim.time + dt)
    rate_for.append(r.kinetics.forward_rates_of_progress)
    rate_rev.append(r.kinetics.reverse_rates_of_progress)
    rate_net.append(r.kinetics.net_rates_of_progress)
    delta_H.append(r.kinetics.delta_enthalpy)

  return np.array(rate_for), np.array(rate_rev), np.array(
    rate_net), np.array(delta_H)


def VariableStepCVReactor(
  cfg: ml_collections.ConfigDict, 
  init_T: float, init_P: float, phi: Union[float, dict], 
) -> float:
  """Variable step constant volume reactor to generate ignition delay times for 
  conditions of reduced chemical kinetic mechanisms.

  Args:
    cfg: Config dict that stores mechanism and condition information.
    init_T: Initial temperature of 0D reactor.
    init_P: Initial pressure of 0D reactor.
    phi: Equivalence ratio (float) or mass fraction (list) of 0D reactor.

  Returns:
    idt: Ignition delaty time, in [s].
  """
  t = 0.
  # set the gas
  gas = ct.Solution(cfg.mechpath)
  gas.TP = init_T, init_P * ct.one_atm
  fuel = cfg.fuel if type(cfg.fuel) is str else dict(cfg.fuel)
  ox = cfg.ox if type(cfg.ox) is str else dict(cfg.ox)
  if type(phi) is float:
    gas.set_equivalence_ratio(phi=phi, fuel=fuel, oxidizer=ox)
  else:
    gas.Y = phi

  # set the reactor
  r = ct.IdealGasMoleReactor(contents=gas)
  sim = ct.ReactorNet([r])
  # precon = ct.AdaptivePreconditioner()  # Create the preconditioner, may introduce error
  # sim.preconditioner = precon  # Add it to the network
  sim.verbose = False

  # compute IDT
  while t < cfg.idt_est:
    t = sim.step()
    if r.T - init_T > cfg.ign_rise: # We use the temperatrure increase criteria.
      idt = t
      break

  return idt


def CPReactorSim(
  cfg: ml_collections.ConfigDict, 
  init_T: float, init_P: float, phi: Union[float, dict], 
  sim_time: float,
  red_mech: str=None,
) -> ct.SolutionArray:
  """Variable step constant pressure reactor to generate thermo state evolution 
  during ignition for validating the details of reduced mechanisms.

  Args:
    cfg: Config dict that stores mechanism and condition information.
    init_T: Initial temperature of 0D reactor.
    init_P: Initial pressure of 0D reactor.
    phi: Equivalence ratio (float) or mass fraction (list) of 0D reactor.
    sime_time: Simulation time for the ignition, default 2 * IDT.
    red_mech: Reduced mechanism file, if None, simulate with detailed mechanism.

  Returns:
    states: Thermo state evolution during ignition.
  """
  t = 0.
  # set the gas
  if red_mech is None:
    gas = ct.Solution(cfg.mechpath)
  else:
    gas = ct.Solution(red_mech)
  gas.TP = init_T, init_P * ct.one_atm
  fuel = cfg.fuel if type(cfg.fuel) is str else dict(cfg.fuel)
  ox = cfg.ox if type(cfg.ox) is str else dict(cfg.ox)
  if type(phi) is float:
    gas.set_equivalence_ratio(phi=phi, fuel=fuel, oxidizer=ox)
  else:
    gas.Y = phi

  # set the reactor
  r = ct.IdealGasConstPressureMoleReactor (contents=gas)
  sim = ct.ReactorNet([r])
  precon = ct.AdaptivePreconditioner()  # Create the preconditioner
  sim.preconditioner = precon  # Add it to the network
  sim.verbose = False
  states = ct.SolutionArray(gas, extra=['t'])

  # compute ignition process
  while t < sim_time:
    t = sim.step()
    states.append(r.thermo.state, t=sim.time*1e6)

  return states


