# Standard imports
import torch
from pyrsistent._pmap import PMap
from pyrsistent._plist import PList, _EmptyPList as EmptyPList
from pyrsistent import pmap, plist

def vector(*x):
    # This needs to support both lists and vectors
    try:
        result = torch.stack(x) # NOTE: Important to use stack rather than torch.tensor
    except: # NOTE: This except is horrible, but necessary for list/vector ambiguity
        result = plist(x)
    return result


def hashmap(*x):
    # This is a dictionary
    keys, values = x[0::2], x[1::2]
    checked_keys = []
    for key in keys: # Torch tensors cannot be dictionary keys, so convert here
        if type(key) is torch.Tensor: key = float(key)
        checked_keys.append(key)
    dictionary = dict(zip(checked_keys, values))
    hashmap = pmap(dictionary)
    return hashmap


def push_address(*x):
    # Concatenate two addresses to produce a new, unique address
    previous_address, current_addreess, continuation = x[0], x[1], x[2]
    return continuation(previous_address+'-'+current_addreess)

# blatantly taken from FOPPL assignments.

def first(vec):
    return vec[0]


def second(vec):
    return vec[1]


def rest(vec):
    return vec[1:]


def last(vec):
    return vec[-1]


def append(vec, val):
    if isinstance(vec, torch.Tensor):
        vec = torch.cat([vec, val[None]])
    elif isinstance(vec, PList):
        if empty(vec):
            vec = plist([val])
        else:
            vec = vec.append(val)
    elif empty(vec):
        vec = vector(val)
    else:
        print(type(vec))
        raise ValueError("Unkown vec type.")
    return vec


def nth(idx, vec):
    return vec[idx]


def conj(vec, elem):
    return cons(elem, vec)


def cons(elem, vec):
    if isinstance(vec, torch.Tensor):
        return torch.cat([elem[None], vec])
    else:
        elems = [elem] + [x for x in vec]
        return vector(*elems)


def get(vec, idx):
    if isinstance(idx, str):
        return vec[idx]
    return vec[idx.int().item()]

def put(obj, idx, val):
    if isinstance(obj, PMap) or isinstance(obj, PList):
        if isinstance(idx, str):
            obj = obj.set(idx, val)
        else:
            obj = obj.set(idx.int().item(), val)
    else:
        obj[idx.int().item()] = val
    return obj

def repmat(vec, *dims):
    return torch.tile(vec, [dim.int() for dim in dims])

def remove(vec, index):
    part_1 = vec[:index]
    part_2 = vec[index+1:]
    return conj(part_1, part_2)


def empty(obj):
    return len(obj) == 0

FOPPL_PRIMITIVES = {

    # Comparisons
    '<': torch.lt,
    '<=': torch.le,
    '>': torch.gt,
    '>=': torch.ge,
    '=': torch.eq,
    'and': torch.logical_and,
    'or': torch.logical_or,
    'not': torch.logical_not,
    'first': first,
    'second': second,
    'last': last,
    'peek': first,
    'rest': rest,
    'nth': nth,
    'conj': conj,
    'cons': cons,
    'get': get,
    'put': put,
    'append': append,
    'remove': remove,
    'empty?': empty,
    'mat-transpose': lambda mat: mat.T,
    'mat-repmat': repmat,
    'mat-add': torch.add,
    'mat-tanh': torch.tanh,
    # ...

    # Math
    '+': torch.add,
    '-': torch.sub,
    '*': torch.mul,
    '/': torch.div,
    'sqrt': torch.sqrt,
    'abs': torch.abs,
    'log': torch.log,
    # ...

    # Containers
    'vector': vector,
    'hash-map': hashmap,
    # ...

    # Matrices
    'mat-mul': torch.matmul,
    # ...

    # Distributions
    'normal': torch.distributions.Normal,
    'uniform-continuous': torch.distributions.Uniform,
    'beta': torch.distributions.Beta,
    'bernoulli': torch.distributions.Bernoulli,
    'exponential': torch.distributions.Exponential,
    'discrete': torch.distributions.Categorical,
    'gamma': torch.distributions.Gamma,
    'dirichlet': torch.distributions.Dirichlet,
    'flip': torch.distributions.Bernoulli
    # ...

}
# Primative function dictionary
# NOTE: Fill this in

primitives = {
    # HOPPL
    'push-address' : push_address,
}

HOPPL_PRIMITIVES = {
    k: lambda *x, fnc=v: x[-1](fnc(*x[1:-1])) for k,v in FOPPL_PRIMITIVES.items()
}
primitives.update(HOPPL_PRIMITIVES)