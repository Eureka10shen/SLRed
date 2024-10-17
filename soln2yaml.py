# Copyright 2024 Shen Fang, Beihang University. 

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 

import os
import time
import logging
import cantera as ct
from ruamel import yaml

BlockMap = yaml.comments.CommentedMap

def FlowMap(*args, **kwargs):
    m = yaml.comments.CommentedMap(*args, **kwargs)
    m.fa.set_flow_style()
    return m

def FlowList(*args, **kwargs):
    lst = yaml.comments.CommentedSeq(*args, **kwargs)
    lst.fa.set_flow_style()
    return lst

def write(solution, input_filename='', output_filename='', path='', description='Reduced mechanism from sparse learning method.'):
    """Function to write cantera solution object to cti file.

    Parameters
    ----------
    solution : cantera.Solution
        Model to be written
    output_filename : str, optional
        Name of file to be written; if not provided, use ``solution.name``
    path : str, optional
        Path for writing file.

    Returns
    -------
    output_filename : str
        Name of output model file (.cti)

    Examples
    --------
    >>> gas = cantera.Solution('gri30.cti')
    >>> soln2yaml.write(gas, 'copy_gri30.cti')
    copy_gri30.cti

    """
    if output_filename:
        output_filename = os.path.join(path, output_filename)
    else:
        output_filename = os.path.join(path, f'{solution.name}.yaml')
    if os.path.isfile(output_filename):
        os.remove(output_filename)

    with open(input_filename, 'r') as input_file:
        input_dict = yaml.safe_load(input_file)

    with open(output_filename, 'w') as output_file:
        output_dict = {}
        output_dict['input-files'] = input_filename
        output_dict['units'] = input_dict['units']
        output_dict['phases'] = input_dict['phases']
        output_dict['phases'][0]['elements'] = FlowList(output_dict['phases'][0]['elements'])
        output_dict['phases'][0]['species'] = FlowList([sp.name for sp in solution.species()])

        emitter = yaml.YAML()
        if description:
            emitter.dump(BlockMap([("description", yaml.scalarstring.LiteralScalarString(description))]), output_file)

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

        output_dict['species'] = []
        reduced_species_list = [sp.name for sp in solution.species()]
        species = input_dict.get('species', {})
        for sp in species:
            if sp['name'] in reduced_species_list:
                output_dict['species'].append(sp)

        logging.info(f"Number of species : {len(reduced_species_list):5}")

        species_map = BlockMap([('species', output_dict['species'])])
        species_map.yaml_set_comment_before_after_key('species', before='\n')
        emitter.dump(species_map, output_file)

        output_dict['reactions'] = []
        reduced_reactions_list = [rct.equation for rct in solution.reactions()]
        reactions = input_dict.get('reactions', {})
        gas = ct.Solution(input_filename)
        gas_reactions = gas.reactions()
        idx = 0
        for rct, gas_rct in zip(reactions, gas_reactions):
            if gas_rct.equation in reduced_reactions_list:
                idx += 1
                rct = FlowMap(rct)
                rct.yaml_add_eol_comment('Reaction {}'.format(idx), 'equation')
                output_dict['reactions'].append(rct)

        reactions_map = BlockMap([('reactions', output_dict['reactions'])])
        reactions_map.yaml_set_comment_before_after_key('reactions', before='\n')
        emitter.dump(reactions_map, output_file)

        logging.info(f"Number of reactions : {len(reduced_reactions_list):5}")

