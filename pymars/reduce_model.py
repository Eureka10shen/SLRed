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

"""Module for general model reduction class and function."""
import os
from typing import NamedTuple

import cantera as ct

class ReducedModel(NamedTuple):
  """Represents reduced model and associated metadata
  """
  model: ct.Solution
  filename: str = ''
  error: float = 0.0
  limbo_species: list = []
    

def trim(
  initial_model_file: str, 
  exclusion_list: list, 
  new_model_file: str, 
  phase_name: str='',
) -> ct.Solution:
  """Function to eliminate species and corresponding reactions from model

  Args:
    initial_model_file: Filename for initial model to be reduced.
    exclusion_list: List of species names that will be removed.
    new_model_file: Name of new reduced model file.
    phase_name: Optional name for phase to load from mech file (e.g., 'gas'). 

  Returns:
    new_solution: Model with species and associated reactions eliminated.

  """
  solution = ct.Solution(initial_model_file, phase_name)

  # Remove species if in list to be removed
  final_species = [sp for sp in solution.species() if sp.name not in exclusion_list]
  final_species_names = [sp.name for sp in final_species]

  # Remove reactions that use eliminated species
  final_reactions = []
  for reaction in solution.reactions():
    # remove reactions with an explicit third body that has been removed
    # if hasattr(reaction, 'efficiencies') and not getattr(reaction, 'default_efficiency', 1.0):
    #     if (len(reaction.efficiencies) == 1 and 
    #         list(reaction.efficiencies.keys())[0] in exclusion_list
    #         ):
    #         continue
    reaction_species = list(reaction.products.keys()) + list(reaction.reactants.keys())
    if all([sp in final_species_names for sp in reaction_species]):
      # remove any eliminated species from third-body efficiencies
      # if hasattr(reaction, 'efficiencies'):
      #     reaction.efficiencies = {
      #         sp:val for sp, val in reaction.efficiencies.items() 
      #         if sp in final_species_names
      #         }
      final_reactions.append(reaction)

  # Create new solution based on remaining species and reactions
  new_solution = ct.Solution(
    thermo='ideal-gas', 
    kinetics='gas',
    species=final_species, 
    reactions=final_reactions,)
  new_solution.TP = solution.TP
  if phase_name:
    new_solution.name = phase_name
  else:
    new_solution.name = os.path.splitext(new_model_file)[0]

  return new_solution
