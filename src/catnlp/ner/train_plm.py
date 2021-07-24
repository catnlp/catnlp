# -*- coding: utf-8 -*-

import logging
import math
import os
from pathlib import Path

import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_scheduler,
    set_seed,
)

from .model.albert_tiny import AlbertTinyCrf, AlbertTinySoftmax
from .model.bert import BertBiaffine, BertCrf, BertSoftmax, BertLstmCrf
from .util.data import NerBertDataset, NerBertDataLoader
from .util.split import recover
from .util.score import get_f1
from .util.decode import get_labels


class PlmTrain:
    def __init__(self, config) -> None:
        logger = logging.getLogger(__name__)
        if config.get("output") is not None:
            os.makedirs(config.get("output"), exist_ok=True)

        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        accelerator = Accelerator(cpu=config["cpu"])
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
        file_format = config.get("file_format")
        input_dir = Path(config.get("input"))
        if file_format in ["bio", "bies", "conll"]:
            train_file = input_dir / "train.txt"
            dev_file = input_dir / "dev.txt"
        else:
            train_file = input_dir / "train.json"
            dev_file = input_dir / "dev.json"
        tokenizer = AutoTokenizer.from_pretrained(config.get("model_path"), use_fast=True)
        train_dataset = NerBertDataset(train_file, tokenizer, config.get("max_length"), file_format=file_format, do_lower=config.get("do_lower_case"))
        dev_dataset = NerBertDataset(dev_file, tokenizer, config.get("max_length"), file_format=file_format, do_lower=config.get("do_lower_case"))
        if file_format == "split":
            dev_contents = dev_dataset.get_contents()
            dev_offset_lists = dev_dataset.get_offset_lists()


        # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
        # unique labels.
        label_file = Path(config.get("output")) / "label.txt"
        train_dataset.save_label(label_file)
        label_list = train_dataset.get_label_list()
        label_to_id = train_dataset.get_label_to_id()
        print(label_to_id)
        num_labels = len(label_list)

        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        pretrained_config = AutoConfig.from_pretrained(config.get("model_path"), num_labels=num_labels)

        model_name = config.get("name").lower()
        if model_name == "bert_crf":
            model_func = BertCrf
        elif model_name == "bert_softmax":
            model_func = BertSoftmax
        elif model_name == "bert_lstm_crf":
            model_func = BertLstmCrf
        elif model_name == "bert_biaffine":
            model_func = BertBiaffine
        elif model_name == "albert_tiny_crf":
            model_func = AlbertTinyCrf
        elif model_name == "albert_tiny_softmax":
            model_func = AlbertTinySoftmax
        else:
            raise ValueError
        
        pretrained_config.loss_name = config.get("loss_name")

        model = model_func.from_pretrained(
            config.get("model_path"),
            config=pretrained_config
        )

        # model.resize_token_embeddings(len(tokenizer))

        # Preprocessing the raw_datasets.
        # First we tokenize all the texts.
        train_dataloader = NerBertDataLoader(train_dataset, batch_size=config.get("per_device_train_batch_size"), shuffle=True, drop_last=False)
        dev_dataloader = NerBertDataLoader(dev_dataset, batch_size=config.get("per_device_dev_batch_size"), shuffle=False, drop_last=False)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        weight_decay = config.get("weight_decay")
        model_type = config.get("model_type")
        plm_lr = config.get("plm_lr")
        not_plm_lr = config.get("not_plm_lr")
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and model_type not in n],
                "weight_decay": weight_decay,
                "lr": not_plm_lr
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and model_type not in n],
                "weight_decay": 0.0,
                "lr": not_plm_lr
            },
                        {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and model_type in n],
                "weight_decay": weight_decay,
                "lr": plm_lr
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and model_type in n],
                "weight_decay": 0.0,
                "lr": plm_lr
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=plm_lr)

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
        writer = SummaryWriter(config.get("summary"))
        completed_steps = 0
        best_f1 = 0

        for epoch in range(config.get("num_train_epochs")):
            model.train()
            train_loss = 0.0
            dev_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "label_mask": batch[4], "input_len": batch[5]}
                outputs = model(**inputs)
                loss = outputs
                loss = loss / config.get("gradient_accumulation_steps")
                accelerator.backward(loss)
                train_loss += loss.item()
                if step % config.get("gradient_accumulation_steps") == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                if completed_steps >= config.get("max_train_steps"):
                    break

            model.eval()
            device_type = device.type
            decode_type = config.get("decode_type")
            pred_lists = list()
            gold_lists = list()
            for step, batch in enumerate(dev_dataloader):
                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "label_mask": batch[4], "input_len": batch[5]}
                    outputs = model(**inputs)
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "label_mask": batch[4], "input_len": batch[5]}
                    loss = model(**inputs)
                    dev_loss += loss.item()
                labels = batch[3]
                predictions_gathered = accelerator.gather(outputs)
                labels_gathered = accelerator.gather(labels)
                preds, golds = get_labels(predictions_gathered, labels_gathered, label_list, masks=batch[6], decode_type=decode_type, device=device_type)
                pred_lists += preds
                gold_lists += golds
            
            if file_format == "split":
                new_pred_lists = list()
                new_gold_lists = list()
                start_idx = 0
                for dev_content, dev_offset_list in zip(dev_contents, dev_offset_lists):
                    end_idx = start_idx + len(dev_offset_list)
                    pred_list = recover(dev_content, pred_lists[start_idx: end_idx], dev_offset_list)
                    gold_list = recover(dev_content, gold_lists[start_idx: end_idx], dev_offset_list)
                    new_pred_lists.append(pred_list)
                    new_gold_lists.append(gold_list)
                    start_idx = end_idx
                pred_lists = new_pred_lists
                gold_lists = new_gold_lists
            
            accelerator.print(f"\nepoch: {epoch}")
            f1, table = get_f1(gold_lists, pred_lists, format=file_format)
            writer.add_scalars("f1",
                            {
                                "dev": round(100*f1, 2)
                            },
                            epoch + 1)
            if file_format != "biaffine":
                train_loss /= 100.0
                dev_loss /= 10.0
            else:
                train_loss *= 10.0
                dev_loss *= 10.0
            writer.add_scalars('loss',
                            {
                                "train": round(train_loss, 2),
                                "dev": round(dev_loss, 2)
                            },
                            epoch + 1)
            if f1 > best_f1:
                best_f1 = f1
                print(table)
                output_model = config.get("output")
                output_model_file = os.path.join(output_model, "pytorch_model.bin")
                torch.save(model.state_dict(), output_model_file)
                output_config_file = os.path.join(output_model, "config.json")
                with open(output_config_file, 'w') as f:
                    f.write(model.config.to_json_string())
                # accelerator.wait_for_everyone()
                # unwrapped_model = accelerator.unwrap_model(model)
                # unwrapped_model.save_pretrained(config.get("output"), save_function=accelerator.save)
