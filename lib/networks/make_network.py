import os
import importlib

def make_network(cfg):
    module = cfg.network_module
    if hasattr(cfg, 'network_kwargs'):
        kwargs = cfg.network_kwargs
        network = importlib.import_module(module).Network(**kwargs)
    else:
        network = importlib.import_module(module).Network()
    return network

def make_sub_network(cfg):
    subnets = cfg.sub_networks
    sub_networks = []
    for subnet in subnets:
        module = subnet
        sub_networks.append(importlib.import_module(module).Network())
    return sub_networks

def make_embedder(cfg):
    module = cfg.embedder.module
    kwargs = cfg.embedder.kwargs
    embedder = importlib.import_module(module).Embedder(**kwargs)
    return embedder

def make_viewdir_embedder(cfg):
    module = cfg.viewdir_embedder.module
    kwargs = cfg.viewdir_embedder.kwargs
    embedder = importlib.import_module(module).Embedder(**kwargs)
    return embedder

def make_deformer(cfg):
    module = cfg.tpose_deformer.module
    kwargs = cfg.tpose_deformer.kwargs
    deformer = importlib.import_module(module).Deformer(**kwargs)
    return deformer

def make_residual(cfg):
    if 'color_residual' in cfg:
        module = cfg.color_residual.module
        kwargs = cfg.color_residual.kwargs
        residual = importlib.import_module(module).Residual(**kwargs)
        return residual
    else:
        from lib.networks.residuals.zero_residual import Residual
        return Residual()

def make_color_network(cfg, **kargs):
    if "color_network" in cfg:
        module = cfg.color_network.module
        kwargs = cfg.color_network.kwargs
    elif "network" in cfg and "color" in cfg.network:
        if "module" in cfg.network.color:
            module = cfg.network.color.module
        else:
            module = "lib.networks.nerf.nerf_network"
        kwargs = cfg.network.color 

    full_args = dict(kwargs, **kargs)
    color_network = importlib.import_module(module).ColorNetwork(**full_args)
    return color_network

# def _make_module_factory(cfgname, classname):
#     def make_sth(cfg, **kargs):
#         module = getattr(cfg, cfgname).module
#         kwargs = getattr(cfg, cfgname).kwargs
#         full_args = dict(kwargs, **kargs)
#         sth = importlib.import_module(module).__dict__[classname](**full_args)
#         return sth
#     return make_sth

# possible_modules = [ 
#     {
#         "cfgname": "color_network",
#         "classname": "ColorNetwork",
#     }
# ]