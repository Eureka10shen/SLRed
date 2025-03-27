# The MIT License (MIT)
# 
# Copyright (c) 2016-2019 Parker Clayton, Phillip Mestas, and Kyle Niemeyer
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# Copyright 2024 Shen Fang, Beihang University. 
# Adjustments on the file wrting function to serve for Cantera 3.0.

"""Module containing sensitivity analysis reduction stage. """
import os
import time
import logging
import numpy as np
import cantera as ct

from pymars.sampling import sample_metrics, calculate_error, read_metrics
from pymars.reduce_model import trim, ReducedModel
from utils.soln2yaml import write

# Taken from http://stackoverflow.com/a/22726782/1569494
try:
  from tempfile import TemporaryDirectory
except ImportError:
  from contextlib import contextmanager
  import shutil
  import tempfile
  import errno

  @contextmanager
  def TemporaryDirectory():
    name = tempfile.mkdtemp()
    try:
      yield name
    finally:
      try:
        shutil.rmtree(name)
      except OSError as e:
        # Reraise unless ENOENT: No such file or directory
        # (ok if directory has already been deleted)
        if e.errno != errno.ENOENT:
          raise


def evaluate_species_errors(
  starting_model: ReducedModel, 
  ignition_conditions: list, 
  metrics: np.ndarray, 
  species_limbo: list, 
  phase_name: str='', 
  num_threads: int=1,
) -> np.ndarray:
  """Calculate error induced by removal of each limbo species

  Args:
    starting_model: Container with model and file information.
    ignition_conditions: List of autoignition initial conditions.
    metrics: Calculated metrics for starting model, used for evaluating error.
    species_limbo: List of species to consider removal.
    phase_name: Optional name for phase to load from mech file (e.g., 'gas'). 
    num_threads : Number of CPU threads to use for performing simulations.
  
  Returns:
    species_errors: Maximum errors induced by removal of each limbo species.
  """
  species_errors = np.zeros(len(species_limbo))
  with TemporaryDirectory() as temp_dir:
    for idx, species in enumerate(species_limbo):
      test_model = trim(starting_model.filename, 
                        [species], 
                        f'reduced_model_{species}.yaml', 
                        phase_name=phase_name,)
      test_model_file = os.path.splitext(starting_model.filename)[0] + '_'  +\
        f'{test_model.n_species:d}' + 'sp_DRGEPSA.yaml'
      write(test_model, 
            input_filename=starting_model.filename, 
            output_filename=test_model_file, 
            description='Reduced mechanism from DRGEPSA with pyMARS.')
      reduced_model_metrics = sample_metrics(
        test_model_file, ignition_conditions, phase_name=phase_name, 
        num_threads=num_threads,)
      species_errors[idx] = calculate_error(metrics, reduced_model_metrics)

  return species_errors


def run_sa(
  model_file: str, 
  starting_error: float, 
  ignition_conditions: list, 
  psr_conditions: list, 
  flame_conditions: list,
  error_limit: float, 
  species_safe: list, 
  phase_name: str='', 
  algorithm_type: str='greedy', 
  species_limbo: list=[],
  num_threads: int=1, 
  path: str='',
) -> ReducedModel:
  """Runs a sensitivity analysis to remove species on a given model.
  
  Args:
    model_file: Model being analyzed.
    starting_error: Error percentage between the reduced and original models.
    ignition_conditions: List of autoignition initial conditions.
    psr_conditions: List of PSR simulation conditions.
    flame_conditions: List of laminar flame simulation conditions.
    error_limit: Maximum allowable error level for reduced model.
    species_safe: List of species names to always be retained.
    phase_name: Optional name for phase to load from CTI file (e.g., 'gas'). 
    algorithm_type: Type of sensitivity analysis: 
      initial (order based on initial error), or 
      greedy (all species error re-evaluated after each removal).
    species_limbo: List of species to consider; if empty, consider all not in 
      ``species_safe``.
    num_threads: Number of CPU threads to use for performing simulations.
    path: Optional path for writing files.

  Returns:
  ReducedModel: Return reduced model and associated metadata.

  """
  time0 = time.time()
  current_model = ReducedModel(
    model=ct.Solution(model_file, phase_name), 
    error=starting_error, 
    filename=model_file
  )
  logging.info(f'Beginning sensitivity analysis stage, using {algorithm_type} approach.')

  # The metrics for the starting model need to be determined or read
  initial_metrics = sample_metrics(
    model_file, ignition_conditions, reuse_saved=True, phase_name=phase_name,
    num_threads=num_threads, path=path
    )
  time1 = time.time()
  logging.info(f"DRGEPSA metrices computing time: {time1 - time0:9.4f}.")

  if not species_limbo:
    species_limbo = [
      sp for sp in current_model.model.species_names if sp not in species_safe
    ]

  logging.info(53 * '-')
  logging.info('Number of species |  Species removed  | Max error (%)')

  # Need to first evaluate all induced errors of species; for the ``initial`` method,
  # this will be the only evaluation.
  species_errors = evaluate_species_errors(
    current_model, ignition_conditions, initial_metrics, species_limbo, 
    phase_name=phase_name, num_threads=num_threads
    )
  time2 = time.time()
  logging.info(f"DRGEPSA species-induced errors time: {time2 - time1:9.4f}.")

  # Use a temporary directory to avoid cluttering the working directory with
  # all the temporary model files
  with TemporaryDirectory() as temp_dir:
    while species_limbo:
      # use difference between error and current error to find species to remove
      idx = np.argmin(np.abs(species_errors - current_model.error))
      species_errors = np.delete(species_errors, idx)
      species_remove = species_limbo.pop(idx)

      test_model = trim(current_model.filename, 
                        [species_remove], 
                        f'reduced_model_{species_remove}.yaml', 
                        phase_name=phase_name)
      test_model_file = os.path.splitext(model_file)[0] + '_'  +\
        f'{test_model.n_species:d}' + 'sp_DRGEPSA.yaml'
      write(test_model, 
            input_filename=model_file, 
            output_filename=test_model_file, 
            description='Reduced mechanism from DRGEPSA with pyMARS.')
      reduced_model_metrics = sample_metrics(
        test_model_file, ignition_conditions, phase_name=phase_name, 
        num_threads=num_threads, path=path)
      error = calculate_error(initial_metrics, reduced_model_metrics)

      logging.info(f'{test_model.n_species:^17} | {species_remove:^17} | {error:^.2f}')

      # cleanup files
      if current_model.filename != model_file:
          os.remove(current_model.filename)

      # Ensure new error isn't too high
      if error > error_limit:
        break
      else:
        current_model = ReducedModel(model=test_model, filename=test_model_file, error=error)

      # If using the greedy algorithm, now need to reevaluate all species errors
      if algorithm_type == 'greedy':
        species_errors = evaluate_species_errors(
          current_model, ignition_conditions, initial_metrics, species_limbo, 
          phase_name=phase_name, num_threads=num_threads)
        if min(species_errors) > error_limit:
          break

  # cleanup files
  os.remove(test_model_file)
  time3 = time.time()
  logging.info(f"DRGEPSA evaluation and reduction time: {time3 - time2:9.4f}.")

  # Final model; may need to rewrite
  reduced_model = ReducedModel(model=current_model.model, 
                               error=current_model.error)
  logging.info(53 * '-')
  logging.info('Sensitivity analysis stage complete.')
  logging.info(f'Skeletal model: {reduced_model.model.n_species} species and '
               f'{reduced_model.model.n_reactions} reactions.')
  logging.info(f'Maximum error: {reduced_model.error:.2f}%')

  return reduced_model


