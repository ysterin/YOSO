#!/bin/bash
micromamba run --name yoso accelerate launch  --num_processes=1 --mixed_precision=fp16 "$@"