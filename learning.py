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

from absl import logging
import time
import ml_collections
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from plot import plot_record, plot_weight


def get_dataset(xdot: np.ndarray, rate: np.ndarray, batch_size: int):
  """Make dataset as a torch dataloader.
  """
  rate = torch.from_numpy(rate)
  xdot = torch.from_numpy(xdot)
  dataset = TensorDataset(rate, xdot)
  data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=8)

  return data_loader


class sparse_model(nn.Module):
  def __init__(self, mat):
    super().__init__()
    self.mat = nn.Parameter(mat)
    self.sparse_weight = torch.Tensor(mat.shape[0]).uniform_(1, 2)
    self.sparse_weight = nn.Parameter(self.sparse_weight)

  def forward(self, rate):
    x = rate * F.sigmoid(self.sparse_weight) @ self.mat 
    return x


def train(
  cfg: ml_collections.ConfigDict, 
  spe_rates: np.ndarray, 
  rct_rates: np.ndarray, 
  mat: np.ndarray,
  clip: float=1.e-1,
) -> np.ndarray:
  """Sparse learning reduced mechanisms by optimizing the weights.
  """
  # set parallel dataset, model and optimizer
  sl_dataloader = get_dataset(spe_rates, rct_rates, cfg.batch_size)
  model = sparse_model(torch.from_numpy(mat)).to('cuda')
  optim = torch.optim.Adam([model.sparse_weight], lr=cfg.lr)
  loss_fn = nn.MSELoss()

  # optimizing
  loss_record = np.zeros((cfg.n_epochs, 3))
  start_optim = time.time()
  for epoch in range(cfg.n_epochs):
    optim.zero_grad()
    for _, (rate, x_dot) in enumerate(sl_dataloader):
      rate = rate.to('cuda')
      x_dot = x_dot.to('cuda')
      output = model(rate)

      norm_factor = x_dot.abs().max(dim=1, keepdim=True)[0].clip(min=clip)
      regression_loss = ((output - x_dot) / norm_factor).norm(p=2, dim=1).mean()
      sparse_loss = (F.sigmoid(model.sparse_weight)).mean()
      loss = (1 - cfg.lambda_1) * regression_loss + cfg.lambda_1 * sparse_loss

      loss.backward()
      optim.step()

      # logging
      loss_record[epoch, 0] += regression_loss.item()
      loss_record[epoch, 1] += sparse_loss.item()
      loss_record[epoch, 2] += loss.item()
    if (epoch + 1) % cfg.log_epoch == 0:
      logging.info(f"Epoch : {epoch:4}, loss : {loss.item():>10.5f}, "
                   f"regression loss : {regression_loss.item():>10.5f}, "
                   f"sparse loss : {sparse_loss.item():>10.5f}, "
                   f"time : {time.time()-start_optim:>10.5f}.")
      weight = F.sigmoid(model.sparse_weight).data.cpu().numpy()
      np.save('./data/' + cfg.mech + '/sparse_weight.npy', weight)
      np.save('./data/' + cfg.mech + '/loss_record.npy', loss_record)
      plot_weight(weight, cfg.mech)
      plot_record(loss_record, epoch, cfg.mech)

  logging.info("#"*60)
  logging.info("# Sparse learning done !")
  logging.info("#"*60)

  return weight


