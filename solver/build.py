# encoding: utf-8
"""
@author: mikwieczorek
"""

import torch


def build_optimizer(named_parameters, hparams):
    regular_parameters = []
    regular_parameter_names = []

    center_parameters = []
    center_parameters_names = []

    for name, parameter in named_parameters:
        if parameter.requires_grad is False:
            print(f'Parameter {name} does not need a Grad. Excluding from the optimizer...')
            continue
        elif 'center' in name:
            center_parameters.append(parameter)
            center_parameters_names.append(name)
        else:
            regular_parameters.append(parameter)
            regular_parameter_names.append(name)

    param_groups = [
        {"params": regular_parameters, "names": regular_parameter_names},
    ]

    center_params_group = [
        {'params': center_parameters, "names": center_parameters_names}
    ]

    if hparams.SOLVER.OPTIMIZER_NAME == "Adam":
        optimizer = torch.optim.Adam
        model_optimizer = optimizer(
        param_groups, lr=hparams.SOLVER.BASE_LR, weight_decay=hparams.SOLVER.WEIGHT_DECAY,
    )
    else:
        raise NotImplementedError(f"No such optimizer {hparams.SOLVER.OPTIMIZER_NAME}")

    optimizers_list = [model_optimizer]
    optimizer_center = torch.optim.SGD(center_params_group, lr=hparams.SOLVER.CENTER_LR)
    optimizers_list.append(optimizer_center)

    return optimizers_list


def build_scheduler(model_optimizer, hparams):
    if hparams.SOLVER.LR_SCHEDULER_NAME == "cosine_annealing":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                model_optimizer, hparams.SOLVER.MAX_EPOCHS, eta_min=hparams.SOLVER.MIN_LR
            )
    elif hparams.SOLVER.LR_SCHEDULER_NAME == "multistep_lr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            model_optimizer, milestones=hparams.SOLVER.LR_STEPS,
            gamma=hparams.SOLVER.GAMMA
        )
    else:
        raise NotImplementedError(f"No such scheduler {hparams.SOLVER.LR_SCHEDULER_NAME}")

    return lr_scheduler