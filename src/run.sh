#!/bin/bash
for ((repeat=1;repeat<=5;repeat++))
do
	echo "repeat: $repeat"
	python main_version.py --test --version 0 --device 1 --n_epochs 100 --beta 0.9 --source CD --target Movie --lr 0.0001 --hidden 64 64  --user_emb_mode cml --out_dim_encode 64
done
