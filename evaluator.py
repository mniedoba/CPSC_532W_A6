# Standard imports
import torch
from enum import Enum
from time import time
from uuid import uuid4
from tqdm import trange

# Project imports
from primitives import primitives

class ExpressionType(Enum):

    SYMBOL = 0
    CONSTANT = 1
    IF_BLOCK = 2
    EXPR_LIST = 3
    SAMPLE = 4
    OBSERVE = 5
    FUNCTION = 6

    @classmethod
    def parse_type(cls, expr):

        if isinstance(expr, str) and expr[0] != "\"" and expr[0] != "\'":
            return ExpressionType.SYMBOL
        elif not isinstance(expr, list):
            return ExpressionType.CONSTANT
        elif expr[0] == 'sample':
            return ExpressionType.SAMPLE
        elif expr[0] == 'observe':
            return ExpressionType.OBSERVE
        elif expr[0] == 'fn':
            return ExpressionType.FUNCTION
        elif expr[0] == 'if':
            return ExpressionType.IF_BLOCK
        else:
            return ExpressionType.EXPR_LIST


class Env(dict):
    'An environment: a dict of {var: val} pairs, with an outer environment'
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        'Get var from the innermost env.'
        if var in self:
            result = self[var]
        elif var[:4] == 'addr':
            return var
        else:
            if self.outer is None:
                raise ValueError(f'Outer limit of environment reached, {var} not found.')
            else:
                result = self.outer.find(var)
        return result

class Procedure(object):
    'A user-defined HOPPL procedure'
    def __init__(self, params:list, body:list, sig:dict, env:Env):
        self.params = params
        self.body = body
        self.sig = sig
        self.env = env
    def __call__(self, *args):
        return eval(self.body, self.sig, Env(self.params, args, self.env))

    def reset_weight(self):
        self.sig['logW'] = 0.


def standard_env():
    'An environment with some standard procedures'
    env = Env()
    env.update(primitives)
    return env


def eval(e, sig:dict, env:Env, verbose=False):
    '''
    The eval routine
    @params
        e: expression
        sig: side-effects
        env: environment
    '''
    expr_type = ExpressionType.parse_type(e)
    match expr_type:
        case ExpressionType.SYMBOL:
            return env.find(e)
        case ExpressionType.CONSTANT:
            if isinstance(e, str):
                return e
            return torch.tensor(e, dtype=torch.float)
        case ExpressionType.IF_BLOCK:
            _, pred, cons, ante = e
            if eval(pred, sig, env):
                return eval(cons, sig, env)
            else:
                return eval(ante, sig, env)
        case ExpressionType.EXPR_LIST:
            evaluated = []
            for sub_expr in e:
                evaluated_sub_expr = eval(sub_expr, sig, env)
                evaluated.append(evaluated_sub_expr)
            proc = evaluated[0]
            args = evaluated[1:]
            return proc(*args)
        case ExpressionType.SAMPLE:
            _, addr_expr, dist_expr, cont = e
            new_alpha = eval(addr_expr, sig, env)
            sig['address'] = new_alpha
            # env = Env(['alpha'], [new_alpha], outer=env)
            dist = eval(dist_expr, sig, env)
            sample = dist.sample()
            return eval(cont, sig, env), [sample], sig, False
        case ExpressionType.OBSERVE:
            _, addr_expr, dist_expr, obs_expr, cont = e
            new_alpha = eval(addr_expr, sig, env)
            sig['address'] = new_alpha
            dist = eval(dist_expr, sig, env)
            obs = eval(obs_expr, sig, env)
            sig['logW'] += dist.log_prob(obs)
            return eval(cont, sig, env), [obs], sig, True
        case ExpressionType.FUNCTION:
            _, params, body = e
            return Procedure(params, body, sig, env)
    return


def evaluate(ast:dict, sig=None, run_name='start', verbose=False):
    '''
    Evaluate a HOPPL program as desugared by daphne
    Args:
        ast: abstract syntax tree
    Returns: The return value of the program
    '''
    if sig is None:
        sig = {}
    sig['logW'] = 0.
    env = standard_env()
    output = lambda x: x # Identity function, so that output value is identical to output
    exp = eval(ast, sig, env, verbose)(run_name, output) # NOTE: Must run as function with a continuation
    while type(exp) is tuple: # If there are continuations the exp will be a tuple and a re-evaluation needs to occur
        func, args, sig, is_obs = exp
        exp = func(*args)
    return exp, sig