#!/bin/bash

python run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=250 --seed=0 --timesteps_per_batch=1200  --env=WindTurbine-v0 --video=0 --plot #--outfile=outdir/WindTurbine-v0.h5
