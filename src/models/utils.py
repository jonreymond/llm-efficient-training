import torch
from .llama import Llama
from .base import GPTBase, LayerNorm

from .noam import Noam

from .common import RMSNorm





BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    LayerNorm,
    RMSNorm,
    torch.nn.Embedding,
)


def get_model(args):
    """ Return the right model """
    if args.model == 'base':
        model = GPTBase(args)
        return model
    elif args.model == 'llama2':
        model = Llama(args)
        return model
    elif args.model == "noam":
        print("here")
        model = Noam(args)
    else:
        raise KeyError(f"Unknown model '{args.model}'.")
    
    return model
