# Standard imports
import copy
import math
from multiprocessing import Pool
from time import time
from tqdm import trange, tqdm
import torch
import numpy as np
import sys

# Project imports
from evaluator import eval, evaluate, standard_env
from utils import log_sample_to_wandb, log_samples_to_wandb
from utils import resample_using_importance_weights, check_addresses, calculate_effective_sample_size

def get_samples(ast:dict, num_samples:int, tmax=None, inference=None, wandb_name=None, verbose=False):
    '''
    Get some samples from a HOPPL program
    '''
    if inference is None:
        samples = get_prior_samples(ast, num_samples, tmax, wandb_name, verbose)
    elif inference == 'IS':
        samples = get_importance_samples(ast, num_samples, tmax, wandb_name, verbose)
    elif inference == 'SMC':
        samples = get_SMC_samples(ast, num_samples, tmax, wandb_name, verbose)
    else:
        print('Inference scheme:', inference, type(inference))
        raise ValueError('Inference scheme not recognised')
    return samples


def get_prior_samples(ast:dict, num_samples:int, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of samples from the prior of a HOPPL program
    '''
    samples = []
    if (tmax is not None): max_time = time()+tmax
    for i in trange(num_samples):
        sample, _ = evaluate(ast, verbose=verbose)
        if wandb_name is not None: log_sample_to_wandb(sample, i, wandb_name=wandb_name)
        samples.append(sample)
        if (tmax is not None) and (time() > max_time): break
    return samples


def get_importance_samples(ast:dict, num_samples:int, tmax=1E6, wandb_name=None, verbose=False):
    '''
    Generate a set of importamnce samples from a HOPPL program
    '''
    samples, log_weights = [], []
    if tmax is not None:
        max_time = time() + tmax
    for i in trange(num_samples):
        sample, sig = evaluate(ast, verbose=verbose)
        samples.append(sample)
        log_weights.append(sig['logW'])
        if (tmax is not None) and (time() > max_time):
            break
    samples = resample_using_importance_weights(samples, log_weights)
    return samples

def run_forward(f, a):
    done = False
    fnc = f
    args = a
    is_obs=False
    while not done:
        result = fnc(*args)
        if type(result) is tuple:
            fnc, args, sig, is_obs = result
            done = is_obs
        else:
            done = True
    return result


def return_fnc(x):
    return x

def resample(particles, weights):
    """Resample particles according to weights"""
    resampled_particles = []
    n_particles = len(particles)
    sum_weights = torch.cumsum(weights, dim=0)
    start = np.random.random() / n_particles
    particle_idx = 0
    for i in range(n_particles):
        weight = start + i / n_particles
        while sum_weights[particle_idx] < weight:
            particle_idx += 1
        resampled_particles.append(copy.deepcopy(particles[i]))
    for particle in resampled_particles:
        particle[0].reset_weight()
    return resampled_particles

def get_SMC_samples(ast:dict, num_samples:int, tmax=None, run_name='start', wandb_name=None, verbose=False):
    '''
    Generate a set of samples via Sequential Monte Carlo from a HOPPL program
    '''
    start_fnc = eval(ast, {'logW': 0.}, standard_env(), verbose)
    args = ("start", return_fnc)
    particles = []
    for i in trange(num_samples, leave=False):
        start_fnc.sig = {'logW': 0.}
        particles.append(run_forward(start_fnc, args))
    obs_number = 0
    total_log_evidence = torch.zeros(1)
    log_weights = torch.zeros(num_samples)
    while type(particles[0]) is tuple:
        obs_number += 1
        log_weights = torch.tensor([particle[2]['logW'] for particle in particles])
        weights = torch.exp(log_weights)
        avg_weight = weights.mean()
        log_evidence = torch.log(avg_weight)
        total_log_evidence += log_evidence
        ESS = calculate_effective_sample_size(torch.exp(log_weights).type(torch.float64))
        print(f"Observation {obs_number}. Observation log evidence: {log_evidence}, ESS: {ESS}")
        if ESS < num_samples / 2.:
            print(f'Resampling.')
            particles = resample_using_importance_weights(particles, log_weights)
            for i in range(num_samples):
                particles[i] = copy.deepcopy(particles[i])
                particles[i][0].sig['logW'] = 0.
                # Maybe deepcopy?
        for i, particle in enumerate(tqdm(particles, leave=False)):
            fnc, args = particle[:2]
            particles[i] = run_forward(fnc, args)

    print(f"Program complete. Total Log Evidence: {total_log_evidence}.")
    samples = particles
    return resample_using_importance_weights(samples, log_weights)


