#!/bin/bash

python run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=500 --seed=0 --timesteps_per_batch=14400  --env=WindTurbineStepwise-v0 --video=0 --outfile=outdir/WindTurbineStepwise-v0_energy17.h5 --use_hd=1 --snapshot_every=50
#python run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=500 --seed=0 --timesteps_per_batch=14400  --env=WindTurbine-v0 --video=0 --outfile=outdir/WindTurbine-v0_energy13.h5 --use_hd=1 --snapshot_every=50
#python run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=250 --seed=0 --timesteps_per_batch=5000  --env=CartPole-v0 --video=0 #--plot -outfile=outdir/WindTurbine-v0.h5


python run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=500 --seed=0 --timesteps_per_batch=14400  --env=WindTurbineStepwise-v0 --video=0 --outfile=outdir/WindTurbineStepwise-v0_energy17.h5 --use_hd=1 --snapshot_every=50


# --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=10 --seed=0 --timesteps_per_batch=900  --env=WindTurbine-v0 --video=0 --outfile=WindTurbine-v0_energy.h5 --use_hd=1 --snapshot_every=5