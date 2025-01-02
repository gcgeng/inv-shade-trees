import os
import importlib

def make_visualizer(cfg, name=None):
    module = cfg.visualizer_module
    if name is None and cfg.vis_name is not "default":
        name = cfg.vis_name
    visualizer = importlib.import_module(module).Visualizer(name)
    return visualizer
