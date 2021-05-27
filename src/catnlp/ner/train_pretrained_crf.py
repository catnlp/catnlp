# -*- coding: utf-8 -*-

# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""
import logging
import math
import os
from pathlib import Path

import torch
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator
from datasets import load_metric
from transformers import (
    AdamW,
    AutoConfig,
    get_scheduler,
    set_seed,
)

from .model.albert_tiny import AlbertTinyCrf
from .model.bert import BertCrf
from .util.data import NerBertDataset, NerBertDataLoader
from .util.tokenizer import NerBertTokenizer


logger = logging.getLogger(__name__)


class PretrainedCrfTrain:
    def __init__(self, config) -> None:
        if config.get("output") is not None:
            os.makedirs(config.get("output"), exist_ok=True)

        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        accelerator = Accelerator()
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state)

        # Setup logging, we only want one process per machine to log things on the screen.
        # accelerator.is_local_main_process is only True for one process per machine.
        logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if config.get("seed") is not None:
            set_seed(config.get("seed"))

        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
        # (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
        # 'tokens' is found. You can easily tweak this behavior (see below).
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        input_dir = Path(config.get("input"))
        train_file = input_dir / "train.txt"
        dev_file = input_dir / "dev.txt"
        vocab_file = Path(config.get("model_path")) / "vocab.txt"
        tokenizer = NerBertTokenizer(vocab_file, do_lower_case=config.get("do_lower_case"))
        train_dataset = NerBertDataset(train_file, tokenizer, config.get("max_length"))
        dev_dataset = NerBertDataset(dev_file, tokenizer, config.get("max_length"))


        # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
        # unique labels.
        label_list = train_dataset.get_label_list()
        label_to_id = train_dataset.get_label_to_id()
        num_labels = len(label_list)

        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        pretrained_config = AutoConfig.from_pretrained(config.get("model_path"), num_labels=num_labels)

        model_func = None
        model_name = config.get("name").lower()
        if model_name == "bert_crf":
            model_func = BertCrf
        elif model_name == "albert_tiny_crf":
            model_func = AlbertTinyCrf

        model = model_func.from_pretrained(
            config.get("model_path"),
            config=pretrained_config,
            label_size=len(label_list)
        )

        # model.resize_token_embeddings(len(tokenizer))

        # Preprocessing the raw_datasets.
        # First we tokenize all the texts.
        padding = "max_length" if config.get("pad_to_max_length") else False

        train_dataloader = NerBertDataLoader(train_dataset, batch_size=config.get("per_device_train_batch_size"), shuffle=True, drop_last=True)
        dev_dataloader = NerBertDataLoader(dev_dataset, batch_size=config.get("per_device_dev_batch_size"), shuffle=False, drop_last=False)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.get("weight_decay"),
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.get("learning_rate"))

        # Use the device given by the `accelerator` object.
        device = accelerator.device
        model.to(device)

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, dev_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, dev_dataloader
        )

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.get("gradient_accumulation_steps"))
        config["max_train_steps"] = config.get("num_train_epochs") * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            name=config.get("lr_scheduler_type"),
            optimizer=optimizer,
            num_warmup_steps=config.get("num_warmup_steps"),
            num_training_steps=config.get("max_train_steps"),
        )

        # Metrics
        metric = load_metric("seqeval")

        def get_labels(predictions, references):
            # Transform predictions and references tensos to numpy arrays
            if device.type == "cpu":
                y_pred = predictions.detach().clone().numpy()
                y_true = references.detach().clone().numpy()
            else:
                y_pred = predictions.detach().cpu().clone().numpy()
                y_true = references.detach().cpu().clone().numpy()

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(pred, gold_label) if l > 1]
                for pred, gold_label in zip(y_pred, y_true)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(pred, gold_label) if l > 1]
                for pred, gold_label in zip(y_pred, y_true)
            ]
            return true_predictions, true_labels

        def compute_metrics():
            results = metric.compute()
            if config.get("return_entity_level_metrics"):
                # Unpack nested dictionaries
                final_results = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        for n, v in value.items():
                            final_results[f"{key}_{n}"] = v
                    else:
                        final_results[key] = value
                return final_results
            else:
                return {
                    "precision": results["overall_precision"],
                    "recall": results["overall_recall"],
                    "f1": results["overall_f1"],
                    "accuracy": results["overall_accuracy"],
                }

        # Train!
        total_batch_size = config.get("per_device_train_batch_size") * accelerator.num_processes * config.get("gradient_accumulation_steps")

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {config.get('num_train_epochs')}")
        logger.info(f"  Instantaneous batch size per device = {config.get('per_device_train_batch_size')}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {config.get('gradient_accumulation_steps')}")
        logger.info(f"  Total optimization steps = {config.get('max_train_steps')}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(config.get("max_train_steps")), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        for epoch in range(config.get("num_train_epochs")):
            model.train()
            for step, batch in enumerate(train_dataloader):
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                outputs = model(**inputs)
                loss = outputs
                loss = loss / config.get("gradient_accumulation_steps")
                accelerator.backward(loss)
                if step % config.get("gradient_accumulation_steps") == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= config.get("max_train_steps"):
                    break

            model.eval()
            for step, batch in enumerate(dev_dataloader):
                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                    outputs = model(**inputs)
                predictions = torch.tensor(outputs)
                labels = batch[3]
                if not config.get("pad_to_max_length"):  # necessary to pad predictions and labels for being gathered
                    predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                    labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

                predictions_gathered = accelerator.gather(predictions)
                labels_gathered = accelerator.gather(labels)
                preds, refs = get_labels(predictions_gathered, labels_gathered)
                metric.add_batch(
                    predictions=preds,
                    references=refs,
                )  # predictions and preferences are expected to be a nested list of labels, not label_ids

            # eval_metric = metric.compute()
            eval_metric = compute_metrics()
            accelerator.print(f"epoch {epoch}:", eval_metric)

        if config.get("output") is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(config.get("output"), save_function=accelerator.save)
