#!/bin/bash
docker run -itd --rm --name qwen_vl_2-5-finetune \
-v /mnt/truenas/datasets/synth/bangla_single_line/images:/data/shared/Qwen/qwen-vl-finetune/data/images \
-v ./data:/data/shared/Qwen/qwen-vl-finetune/data \
-v ./qwenvl:/data/shared/Qwen/qwen-vl-finetune/qwenvl \
-v ./scripts/:/data/shared/Qwen/qwen-vl-finetune/scripts \
-v ./output/:/data/shared/Qwen/qwen-vl-finetune/output \
-v ../.cache:/root/.cache \
--gpus '"device=0,1,2"' \
--shm-size=2g \
qwen-vl-finetune:latest bash
