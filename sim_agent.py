#!/usr/bin/env python
"""
Load a snapshotted agent from an hdf5 file and animate it's behavior
"""

import argparse
import cPickle, h5py, numpy as np, time
from collections import defaultdict
import gym
import gym_wind_turbine


def animate_rollout(env, agent, n_timesteps,delay=.01):
    infos = defaultdict(list)
    ob = env.reset()
    if hasattr(agent,"reset"): agent.reset()
    # env.render()
    for i in xrange(n_timesteps):
        ob = agent.obfilt(ob) # array([8.,42.14395073,290.92833608,6.8464, 0.606, 0.])
        a, _info = agent.act(ob)
        (ob, rew, done, info) = env.step(a)
        # env.render()
        for (k,v) in info.items():
            infos[k].append(v)
        infos['ob'].append(ob)
        infos['reward'].append(rew)
        infos['action'].append(a)
        if done:
            print("terminated after %s timesteps"%i)
            break
        time.sleep(delay)
    env.render()
    return infos


def animate(env, agent):
    infos = defaultdict(list)
    ob = env.reset()
    done = False
    while not done:
        ob = agent.obfilt(ob)
        a, _info = agent.act(ob)
        (ob, rew, done, info) = env.step(a)
        infos['ob'].append(ob)
        infos['reward'].append(rew)
        infos['action'].append(a)
    env.render()
    return infos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf")
    parser.add_argument("--timestep_limit",type=int)
    parser.add_argument("--snapname")
    args = parser.parse_args()

    hdf = h5py.File(args.hdf,'r')

    snapnames = hdf['agent_snapshots'].keys()
    print "snapshots:\n",snapnames
    if args.snapname is None:
        snapname = snapnames[-1]
    elif args.snapname not in snapnames:
        raise ValueError("Invalid snapshot name %s"%args.snapname)
    else:
        snapname = args.snapname
    # import pdb; pdb.set_trace()
    env = gym.make(hdf["env_id"].value)
    # env = gym.make("WindTurbineConstant-v0")
    # env = gym.make("WindTurbineStepwise-v0")
    agent = cPickle.loads(hdf['agent_snapshots'][snapname].value)
    agent.stochastic=False

    timestep_limit = args.timestep_limit or env.spec.timestep_limit
    i = 1
    print "Episode,Lenght,Reward,Energy,Thrust Area"
    while True:
        # infos = animate_rollout(env,agent,n_timesteps=timestep_limit+1,
        #     delay=1.0/env.metadata.get('video.frames_per_second', 30))
        infos = animate(env,agent)
        reward = 0.0
        energy = 0.0
        thrust_area = 0.0
        length = 0
        step_size = env.env.dt/3600.0
        for _, P, T, _, _, _ in infos["ob"]:
            energy += P * step_size
            thrust_area += T * step_size
        for (k,v) in infos.items():
                if k.startswith("reward"):
                    reward = np.sum(v)
                    length = len(v)
        #import pdb; pdb.set_trace()
        print "{},{},{},{},{}".format(i, length, reward, energy, thrust_area)
        i = i + 1
        # raw_input("press enter to continue")


if __name__ == "__main__":
    main()
