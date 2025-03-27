"""Contains main driver function for pyMARS program."""
import os
import sys
import logging
from argparse import ArgumentParser
from typing import List, Dict, NamedTuple

import yaml
import cantera as ct

# local imports
from pymars.sampling import sample_metrics, parse_ignition_inputs, parse_psr_inputs, parse_flame_inputs
from pymars.sampling import InputIgnition, InputPSR, InputLaminarFlame
import pymars.soln2cti
from pymars.drgep import run_drgep
# from pymars.drg import run_drg
# from pymars.pfa import run_pfa
from pymars.sensitivity_analysis import run_sa
from pymars.tools import convert

#: Supported reduction methods
METHODS = ['DRG', 'DRGEP', 'PFA']

class ReductionInputs(NamedTuple):
    """Collects inputs for overall reduction process.
    """
    model: str
    error: float
    ignition_conditions: List[InputIgnition]
    # psr_conditions: List[InputPSR]
    # flame_conditions: List[InputLaminarFlame]
    method: str
    target_species: List[str]
    safe_species: List[str] = []
    sensitivity_analysis: bool = False
    upper_threshold: float = 0.4
    sensitivity_type: str = 'greedy'
    phase_name: str = ''


def parse_inputs(input_dict):
    """Parses and checks dictionary of inputs for consistency and correctness.

    Parameters
    ----------
    input_dict : dict
        Inputs for reduction

    Returns
    -------
    ReductionInputs
        Object with checked inputs
    
    """
    model = input_dict.get('model', '')
    assert model, 'Input file requires specifying "model".'
    
    error = input_dict.get('error', 0.0)
    assert error, 'Input file requires an error limit specified by "error".'

    method = input_dict.get('method', '')
    sensitivity_analysis = input_dict.get('sensitivity-analysis', False)
    
    assert method or sensitivity_analysis, (
        'Input file requires either "method" or "sensitivity-analysis" to be given.'
        )
    
    if method:
        assert method in METHODS, 'Reduction method must be one of ' + ', '.join(METHODS)

        target_species = input_dict.get('targets', [])
        assert target_species, (
            'At least one "target" species must be specified for graph-based reduction methods.'
            )
    
    upper_threshold = input_dict.get('upper-threshold', None)
    if not upper_threshold and sensitivity_analysis:
        logging.info('Warning: using default upper threshold value (0.1)')
        upper_threshold = 0.1
    sensitivity_type = input_dict.get('sensitivity-type', 'initial')

    safe_species = input_dict.get('retained-species', [])

    phase_name = input_dict.get('phase-name', '')
    
    # check that the specified model actually contains the specified phase
    try:
        gas = ct.Solution(model, phase_name)
    except ValueError:
        raise ValueError(model + ' does not contain phase ' + phase_name)

    # check that species are present in model
    for sp in target_species:
        assert sp in gas.species_names, f'Specified target species {sp} not in model'
    
    for sp in safe_species:
        assert sp in gas.species_names, f'Specified retained species {sp} not in model'
    
    ignition_conditions = input_dict.get('autoignition-conditions', {})
    assert ignition_conditions, 'autoignition-conditions need to be specified'

    # psr_conditions = input_dict.get('psr-conditions', {})
    # flame_conditions = input_dict.get('laminar-flame-conditions', {})
    # if psr_conditions:
    #     raise NotImplementedError('PSR sampling not implemented yet, sorry!')
    # if flame_conditions:
    #     raise NotImplementedError('Laminar flame sampling not implemented yet, sorry!')

    # check validity of input file
    ignition_inputs = parse_ignition_inputs(model, ignition_conditions, phase_name)
    # psr_inputs = parse_psr_inputs(model, psr_conditions, phase_name)
    # flame_inputs = parse_flame_inputs(model, flame_conditions, phase_name)

    return ReductionInputs(
        model=model, error=error, 
        ignition_conditions=ignition_inputs, 
        # psr_conditions=psr_inputs, flame_conditions=flame_inputs,
        method=method, target_species=target_species, safe_species=safe_species,
        sensitivity_analysis=sensitivity_analysis, upper_threshold=upper_threshold,
        sensitivity_type=sensitivity_type, phase_name=phase_name
        )


def main(model_file, error_limit, 
         ignition_conditions, psr_conditions=[], flame_conditions=[],
         method=None, target_species=[], safe_species=[], phase_name='',
         run_sensitivity_analysis=False, upper_threshold=None, sensitivity_type='greedy',
         path='', num_threads=1
         ):
    """Driver function for reducing a chemical kinetic model.

    Parameters
    ----------
    model_file : str
        Cantera-format model to be reduced (e.g., 'mech.cti').
    error_limit : float
        Maximum error percentage for the reduced model.
    ignition_conditions : list of InputIgnition
        List of autoignition initial conditions.
    # psr_conditions : list of InputPSR, optional
    #     List of PSR simulation conditions.
    # flame_conditions : list of InputLaminarFlame, optional
    #     List of laminar flame simulation conditions.
    method : {'DRG', 'DRGEP', 'PFA'}, optional
        Skeletal reduction method to use.
    target_species: list of str, optional
        List of target species for graph-based reduction.
    safe_species : list of str, optional
        List of non-target species to always retain.
    phase_name : str, optional
        Optional name for phase to load from CTI file (e.g., 'gas'). 
    run_sensitivity_analysis : bool, optional
        Flag to run sensitivity analysis after completing another method.
    upper_threshold : float, optional
        Upper threshold (epsilon^*) used to determine species for sensitivity analysis 
        in combination with DRG or DRGEP method
    sensitivity_type : {'initial', 'greedy'}, optional
        Type of sensitivity analysis
    path : str
        Path to directory for writing files
    num_threads : int, optional
        Number of CPU threads to use for performing simulations in parallel.
        Optional; default = 1, in which the multiprocessing module is not used.
        If 0, then use the available number of cores minus one. Otherwise,
        use the specified number of threads.

    """

    if method in ['DRG', 'DRGEP', 'PFA']:
        assert target_species, (
            'Need to specify at least one target species for graph-based reduction methods'
            )
    
    if not method and not run_sensitivity_analysis:
        raise ValueError(
            'Either a graph-based method or sensitivity analysis (or both) must be specified.'
            )
    
    # if method == 'DRG':
    #     reduced_model = run_drg(
    #         model_file, ignition_conditions, psr_conditions, flame_conditions,
    #         error_limit, target_species, safe_species, phase_name=phase_name,
    #         threshold_upper=upper_threshold, num_threads=num_threads, path=path
    #         )    
    if method == 'DRGEP':
        reduced_model = run_drgep(
            model_file, ignition_conditions, psr_conditions, flame_conditions, 
            error_limit, target_species, safe_species, phase_name=phase_name,
            threshold_upper=upper_threshold, num_threads=num_threads, path=path
            )
    # elif method == 'PFA':
    #     reduced_model = run_pfa(
    #         model_file, ignition_conditions, psr_conditions, flame_conditions,
    #         error_limit, target_species, safe_species, phase_name=phase_name,
    #         num_threads=num_threads, path=path
    #         )
    
    error = 0.0
    limbo_species = []
    if method in ['DRG', 'DRGEP', 'PFA']:
        model_file = reduced_model.filename
        error = reduced_model.error
        limbo_species = reduced_model.limbo_species

    if run_sensitivity_analysis:
        if not sensitivity_type:
            sensitivity_type = 'greedy'

        reduced_model = run_sa(
            model_file, error, ignition_conditions, psr_conditions, flame_conditions, 
            error_limit, target_species + safe_species, phase_name=phase_name,
            algorithm_type=sensitivity_type, species_limbo=limbo_species, 
            num_threads=num_threads, path=path
            )

    return reduced_model

if __name__ == '__main__':
    with open('reduction_input.yaml') as the_file:
        input_dict = yaml.safe_load(the_file)
    
    inputs = parse_inputs(input_dict)

    reduced_model = main(
        inputs.model, 
        inputs.error, 
        inputs.ignition_conditions, 
        method=inputs.method, 
        target_species=inputs.target_species,
        safe_species=inputs.safe_species, 
        phase_name=inputs.phase_name,
        run_sensitivity_analysis=inputs.sensitivity_analysis, 
        upper_threshold=inputs.upper_threshold, 
        sensitivity_type=inputs.sensitivity_type, 
        )

    logging.shutdown()
