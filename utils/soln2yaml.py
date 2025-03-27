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
from absl import logging
logging.set_verbosity(logging.DEBUG)
import time
from ruamel import yaml as ryaml
import cantera as ct
from copy import deepcopy

# keys that refer to dicts
DICT_KEYS = ['rate-constant', 'low-P-rate-constant', 'high-P-rate-constant', 
             'Troe', 'efficiencies']

# yaml style helper functions
BlockMap = ryaml.comments.CommentedMap

def FlowMap(*args, **kwargs):
  # formatting dict in yaml
  m = ryaml.comments.CommentedMap(*args, **kwargs)
  m.fa.set_flow_style()
  return m


def FlowList(*args, **kwargs):
  # formatting list in yaml
  lst = ryaml.comments.CommentedSeq(*args, **kwargs)
  lst.fa.set_flow_style()
  return lst


def write(
  solution: ct.Solution, 
  input_filename: str='', 
  output_filename: str='', 
  description: str='Reduced mechanism from SL method.',
):
  """Function to write cantera solution object to yaml file.

  Args:
  solution: Cantera Solution that to be written.
  input_filename: Input mechanism file name.
  output_filename: Output mechanism file name.
  description : Description to be added in the heaad of output mech.
  """
  if os.path.isfile(output_filename):
    os.remove(output_filename)

  with open(input_filename, 'r') as input_file:
    input_dict = ryaml.YAML(typ='safe', pure=True).load(input_file)

  with open(output_filename, 'w') as output_file:
    output_dict = {}
    output_dict['input-files'] = input_filename
    output_dict['units'] = input_dict['units']
    output_dict['phases'] = input_dict['phases']
    output_dict['phases'][0]['elements'] = FlowList(
      set(element for species in solution.species() 
                  for element in solution.species(species.name).composition))
    output_dict['phases'][0]['species'] = FlowList(
      [sp.name for sp in solution.species()])

    emitter = ryaml.YAML()
    if description:
      emitter.dump(BlockMap(
        [("description", ryaml.scalarstring.LiteralScalarString(description))]
      ), output_file)

    # information regarding conversion
    metadata = BlockMap([
      ("generator", "soln2yaml"),
      ("cantera-version", "3.0.0"),
      ("date", time.ctime()),
    ])
    if input_filename != '':
      metadata['input-files'] = input_filename
    if description:
      metadata.yaml_set_comment_before_after_key("generator", before="\n")
    emitter.dump(metadata, output_file)

    units_map = BlockMap([('units', FlowMap(input_dict['units']))])
    units_map.yaml_set_comment_before_after_key('units', before='\n')
    emitter.dump(units_map, output_file)

    phases_map = BlockMap([('phases', output_dict['phases'])])
    phases_map.yaml_set_comment_before_after_key('phases', before='\n')
    emitter.dump(phases_map, output_file)

    # get species formatting
    output_dict['species'] = []
    species_list = [sp.name for sp in solution.species()]
    species = input_dict.get('species', {})
    for sp in species:
      if sp['name'] in species_list:
        sp['thermo']['temperature-ranges'] = FlowList(sp['thermo']['temperature-ranges'])
        for id in range(len(sp['thermo']['data'])):
          sp['thermo']['data'][id] = FlowList(sp['thermo']['data'][id])
        output_dict['species'].append(sp)

    # species formatting
    species_map = BlockMap([('species', output_dict['species'])])
    species_map.yaml_set_comment_before_after_key('species', before='\n')
    emitter.dump(species_map, output_file)

    # get reactions formatting
    output_dict['reactions'] = []
    reactions_list = [rct.equation for rct in solution.reactions()]
    reactions = input_dict.get('reactions', {})
    gas = ct.Solution(input_filename)
    gas_reactions = gas.reactions()
    for rct, gas_rct in zip(reactions, gas_reactions):
      if gas_rct.equation in reactions_list:
        output_dict['reactions'].append(rct)

    # reactions formatting
    for rct in output_dict['reactions']:
      # remove deleted species in three-body reactions
      if 'type' in rct:
        if ((rct['type'] == 'three-body') or (rct['type'] == 'falloff')) \
          and ('efficiencies' in rct.keys()):
          keys = deepcopy(rct['efficiencies']).keys()
          for key in keys:
            if not (key in species_list):
              rct['efficiencies'].pop(key)
        elif rct['type'] == "pressure-dependent-Arrhenius":
          for i in range(len(rct['rate-constants'])):
            rct['rate-constants'][i] = FlowMap(rct['rate-constants'][i])
      # formatting keys
      for key in DICT_KEYS:
        try:
          rct[key] = FlowMap(rct[key])
        except:
          pass

    reactions_map = BlockMap([('reactions', output_dict['reactions'])])
    reactions_map.yaml_set_comment_before_after_key('reactions', before='\n')
    emitter.dump(reactions_map, output_file)


