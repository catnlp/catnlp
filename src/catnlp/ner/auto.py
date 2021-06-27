# -*- coding:utf-8 -*-

import optuna

from .auto_plm import objective


class NerAuto:
    def __init__(self, n_trials):
        print("start auto")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
