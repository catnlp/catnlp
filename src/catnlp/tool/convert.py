# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from ..common.load_file import load_config_file


def convert_tf_checkpoint_to_pytorch(config):
    tf_checkpoint_path = config["tf_checkpoint_path"]
    bert_config_file = config["bert_config_file"]
    pytorch_dump_path = Path(config["pytorch_dump_path"])
    # 初始化pytorch模型
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # 加载tf权重
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # 保持pytorch模型
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换模型格式")
    parser.add_argument(
        "--config", default="data/config/pretrained/bert.yaml", type=str, help="配置文件"
    )
    args = parser.parse_args()

    config = load_config_file(args.config)
    convert_tf_checkpoint_to_pytorch(config)
