# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    LoraArguments
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer, BitsAndBytesConfig

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    # Handle both PEFT and non-PEFT models
    if hasattr(model, 'base_model'):
        # PEFT model (QLoRA)
        base_model = model.base_model.model
    else:
        # Regular model
        base_model = model
    
    if model_args.tune_mm_vision:
        for n, p in base_model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in base_model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in base_model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in base_model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in base_model.model.named_parameters():
            p.requires_grad = True
        base_model.lm_head.requires_grad = True
    else:
        for n, p in base_model.model.named_parameters():
            p.requires_grad = False
        base_model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    print(f"\nLora ARGS: {lora_args}\n")
    lora_config = LoraConfig(
        r=lora_args.r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type=lora_args.task_type
    )

    if data_args.data_packing:
        assert data_args.data_flatten, "data_packing requires data_flatten to be enabled."

    if data_args.data_flatten or data_args.data_packing:
        assert attn_implementation == "flash_attention_2", \
            "data_flatten and data_packing only support 'flash_attention_2' implementation. " \
            f"Current implementation: '{attn_implementation}'."

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else torch.float16),
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        )
        # Adding peft
        print(f"\n\nModel:\n{model}\n\n")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, lora_config)
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # Note: With QLoRA, PEFT automatically manages which parameters are trainable
    # The set_model function is mainly for full fine-tuning scenarios
    # For QLoRA, LoRA target_modules configuration controls what gets trained
    if not hasattr(model, 'base_model'):
        # Only apply set_model for non-PEFT models (full fine-tuning)
        set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.base_model.model.visual.print_trainable_parameters()
        model.base_model.model.model.print_trainable_parameters()
        
        # Print detailed trainable parameter info
        # print("\n=== DETAILED TRAINABLE PARAMETERS ===")
        total_params = 0
        trainable_params = 0
        vision_trainable = 0
        llm_trainable = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'visual' in name or 'vision' in name:
                    vision_trainable += param.numel()
                    # print(f"VISION TRAINABLE: {name} - {param.numel():,} params")
                elif 'model.layers' in name or 'lm_head' in name:
                    llm_trainable += param.numel()
                    # print(f"LLM TRAINABLE: {name} - {param.numel():,} params")
                # else:
                #     print(f"OTHER TRAINABLE: {name} - {param.numel():,} params")
        
        print(f"\n=== SUMMARY ===")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"Vision trainable: {vision_trainable:,} ({100*vision_trainable/trainable_params:.2f}% of trainable)")
        print(f"LLM trainable: {llm_trainable:,} ({100*llm_trainable/trainable_params:.2f}% of trainable)")
        print("=" * 50)
    
    if data_args.data_packing:
        logging.info("Using data packing module")
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        logging.info("Using make_supervised_data_module (not packed)")
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    subset = data_module["train_dataset"]
    underlying_dataset = subset.dataset
    first_idx = subset.indices[0]
    print("--- First Training Sample (RAW) ---")
    if hasattr(underlying_dataset, "list_data_dict"):
        print(underlying_dataset.list_data_dict[first_idx])
    else:
        print("Raw data attribute not found in underlying dataset.")
    print("-----------------------------")

    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="eager")
