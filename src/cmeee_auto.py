# -*- coding:utf-8 -*-

import argparse

from catnlp.ner.auto import NerAuto


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument("--task", type=str,
                        default="NER", help="任务")
    parser.add_argument("--n_trials", type=int,
                        default=8, help="训练配置")
    args = parser.parse_args()

    task = args.task.lower()
    if task == "ner":
        NerAuto(args.n_trials, domain="cmeee")
    else:
        raise RuntimeError(f"{args.task}未开发")
