
from network_modules.vortices.body import body_from_yaml
from network_modules.vortices.yaml_operations import yaml2pyobject

import torch

def load_body_from_parameters(
    pars_path: str,
):
    """ Load body from parameters """

    pars      = yaml2pyobject(pars_path)
    solver    = pars["solver"]
    body_pars = pars["body"]

    device = torch.device("cpu")

    n_steps = solver["N"]+1
    xmin    = solver["xmin"]
    xmax    = solver["xmax"]
    ymin    = solver["ymin"]
    ymax    = solver["ymax"]
    x       = torch.linspace(xmin,xmax,n_steps).to(device)
    y       = torch.linspace(ymin,ymax,n_steps).to(device)
    eps     = 2*float(x[1]-x[0])

    body = body_from_yaml(
        device    = device,
        x         = x,
        y         = y,
        body_pars = body_pars,
        eps       = eps,
    )

    return body

