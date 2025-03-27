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
import numpy as np
import ml_collections
from absl import logging
logging.set_verbosity(logging.INFO)
from copy import deepcopy
from raw_data import getIDTs


def validate(cfg: ml_collections.ConfigDict, red_mech: str):
  """Validate reduced mechanism against user-defined conditions.
  """
  # detailed mech
  idts_path = os.path.join('./data', cfg.mech, 'idts.npy')
  try:
    idts = np.load(idts_path, allow_pickle=True)
  except:
    idts = getIDTs(cfg)

  # reduced mech
  red_cfg = deepcopy(cfg)
  red_cfg.mechpath = red_mech
  idts_red = getIDTs(red_cfg)

  # validate
  error = np.abs(idts - idts_red) / idts * 100.
  # case_idx = np.where(error > 10)[0] # set 10% as the error limit
  # if case_idx.size != 0:
  #   for idx in case_idx:
  #     T_id = int(idx // (len(cfg.init_Ps) * len(cfg.phis)))
  #     left_idx = idx - T_id * (len(cfg.init_Ps) * len(cfg.phis))
  #     P_id = int(left_idx // len(cfg.phis))
  #     phi_id = left_idx - P_id * len(cfg.phis)
  #     logging.info(f"Case {idx:3d}: T_0 = {cfg.init_Ts[T_id]:6.1f}K, "
  #                  f"P_0 = {cfg.init_Ps[P_id]:4.1f}atm, "
  #                  f"phi = {cfg.phis[phi_id]:3.1f}, "
  #                  f"IDT error = {error[idx]:4.1f}%.")
  # else:
  #   logging.info("Very good reduction, no one exceeds the error limit 10%.")

  case_max_error = np.argmax(error)
  T_id = int(case_max_error // (len(cfg.init_Ps) * len(cfg.phis)))
  left_idx = case_max_error - T_id * (len(cfg.init_Ps) * len(cfg.phis))
  P_id = int(left_idx // len(cfg.phis))
  phi_id = left_idx - P_id * len(cfg.phis)
  logging.info(f"Case {case_max_error:3d}: T_0 = {cfg.init_Ts[T_id]:6.1f}K, "
               f"P_0 = {cfg.init_Ps[P_id]:4.1f}atm, "
               f"phi = {cfg.phis[phi_id]:3.1f}, "
               f"IDT error = {error[case_max_error]:4.1f}%.")

  # print(idts_red)

  return error


if __name__ == "__main__":
  from cfgs.propene import default
  cfg = default()
  red_mech = os.path.splitext(cfg.mechpath)[0] + '_76sp.yaml'
  validate(cfg, red_mech)

