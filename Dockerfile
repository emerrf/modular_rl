FROM gym-wind-turbine:latest

RUN pip install theano tabulate keras==1.1.1 h5py

ADD . /app/modular_rl
WORKDIR /app/modular_rl
RUN mkdir mrl_output

ENV KERAS_BACKEND=theano
ENTRYPOINT ["python", "run_pg.py"]
CMD ["--gamma=0.995", "--lam=0.97", "--agent=modular_rl.agentzoo.TrpoAgent", \
"--max_kl=0.01", "--cg_damping=0.1", "--activation=tanh", "--n_iter=250", \
"--seed=0", "--timesteps_per_batch=5000",  "--env=CartPole-v0", "--video=0", \
"--outfile=mrl_output/CartPole-v0.h5"]

# docker build . -t gwt-modular-rl
# docker run -v C:\Users\emerrf\Documents\GitHub\remote\modular_rl\docker_mrl_output:/app/modular_rl/mrl_output gwt-modular-rl --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=10 --seed=0 --timesteps_per_batch=900  --env=WindTurbine-v0 --video=0 --outfile=mrl_output/WindTurbine-v0_energy.h5 --use_hd=1 --snapshot_every=5