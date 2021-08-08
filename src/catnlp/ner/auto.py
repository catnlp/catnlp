# -*- coding:utf-8 -*-

import optuna

from .auto_plm import objective as general_objective
from .auto_plm_cmeee import objective as cmeee_objective


class NerAuto:
    def __init__(self, n_trials, domain):
        print("start auto")
        study = optuna.create_study(direction="maximize")
        if domain == "cmeee":
            objective = cmeee_objective
        else:
            objective = general_objective
        study.optimize(objective, n_trials=n_trials)

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
