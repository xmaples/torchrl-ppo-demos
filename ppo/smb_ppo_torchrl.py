import gym
import gym_super_mario_bros

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
import numpy as np
from torchrl.collectors import MultiSyncDataCollector,SyncDataCollector
from torchrl.data import TensorSpec, UnboundedContinuousTensorSpec,BinaryDiscreteTensorSpec
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchrl.data.replay_buffers import TensorDictPrioritizedReplayBuffer,ReplayBuffer,LazyTensorStorage,SamplerWithoutReplacement,RandomSampler,PrioritizedSampler,TensorDictReplayBuffer
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import TransformedEnv, Compose, ToTensorImage,StepCounter,ObservationNorm, ParallelEnv
import os
from ignite.utils import setup_logger
#  pip install git+https://github.com/xmaples/nes-py.git
from nes_py.wrappers import JoypadSpace

class ReleaseFallingA(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.joypad=self.find_joypad()
        self.falling_A=False
        self.current_y=0
    
    def step(self, action):
        if self.falling_A:
            self.falling_A=False
            action=int(action)
            action=action& ~self.joypad._button_map["A"]
            if action not in self.joypad._action_map.values():
                action=self.joypad._button_map["NOOP"]
            *z, info=self.env.step(action)
            self.current_y=info["y_pos"]
            return *z, info
        
        obs, rew, term, trunc, info = self.env.step(action)
        if info["y_pos"]-self.current_y<0 and self.joypad._action_map[int(action)]&self.joypad._button_map["A"]:
            self.falling_A=True
        else:
            self.falling_A=False
        self.current_y=info["y_pos"]
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        self.falling_A=False
        return self.env.reset(**kwargs)

    def find_joypad(self):
        ptr=self
        while ptr is not None and not isinstance(ptr, JoypadSpace):
            prev_ptr=ptr
            for attr in ["env", "_env","base_env"]:
                if hasattr(ptr, attr):
                    ptr=getattr(ptr, attr)
                    break
            if ptr is prev_ptr:
                ptr=None
        return ptr

class SmbResetInfo(gym.Wrapper):
    
    def reset(self, **kwargs):
        obs, info=self.env.reset(**kwargs)
        if not info:
            info={'coins': 0,
                    'flag_get': False,
                    'life': 2,
                    'score': 0,
                    'status': 'small',
                    'time': 400,
                    'x_pos': 0,
                    'y_pos': 0
                    }
        return obs, info

# smb env return observation and rendered rgb_array with same memory pointer for different steps
class FrameCopy(gym.ObservationWrapper):
    def __init__(self,env):
        super().__init__(env)

    def observation(self, observation):
        return observation.copy()
    
    def render(self,*args, **kwargs):
        if self.render_mode =="rgb_array":
            return super().render(*args, **kwargs).copy()
        else:
            return super().render(*args, **kwargs)
    
    def reset(self, **kwargs):
        obs, info=self.env.reset(**kwargs)
        return obs.copy(), info

class SmbReward(gym.Wrapper):
    def __init__(self, env=None, world=None, stage=None, falling_A_penalty=False):
        super().__init__(env)
        self.current_score = 0
        self.current_x = 0
        self.current_y=0
        self.world = world
        self.stage = stage
        self.falling_A_penalty=falling_A_penalty
        if falling_A_penalty:
            self.joypad=self.find_joypad()
            if self.joypad is None:
                raise ValueError("A Joypad before this is required")

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        self.current_score = info["score"]
        reward /= 100.0
        if done or truncated:
            if info["flag_get"]:
                reward += 1.0
            else:
                reward -= 1.0
                
        if self.falling_A_penalty and info["y_pos"]-self.current_y<0 and self.joypad._action_map[int(action)]&self.joypad._button_map["A"]:
            reward -= 0.1
            
        self.current_x = info["x_pos"]
        self.current_y = info["y_pos"]
        return state, reward, done, truncated, info

    def find_joypad(self):
        ptr=self
        while ptr is not None and not isinstance(ptr, JoypadSpace):
            prev_ptr=ptr
            for attr in ["env", "_env","base_env"]:
                if hasattr(ptr, attr):
                    ptr=getattr(ptr, attr)
                    break
            if ptr is prev_ptr:
                ptr=None
        return ptr

    def reset(self,**kwargs):
        self.current_score = 0
        self.current_x = 0
        self.current_y=0
        return self.env.reset(**kwargs)

class TruncStillX(gym.Wrapper):
    def __init__(self, env, max_steps_still=64):
        super().__init__(env)
        self.stayed_still=0
        self.last_x_pos=0
        self.max_steps_still=max_steps_still
        
    def step(self, action):
        obs,reward,terminated, truncated, info=self.env.step(action)
        done=terminated or truncated
        x_pos=info["x_pos"]
        if self.last_x_pos==x_pos:
            self.stayed_still=self.stayed_still+1
        else:
            self.stayed_still=0
        if not done and self.stayed_still>=self.max_steps_still:
            truncated=True
        self.last_x_pos=x_pos
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.last_x_pos=0
        self.stayed_still=0
        return self.env.reset(**kwargs)

from tensordict.nn import TensorDictModule
from torchrl.modules import ValueOperator, ProbabilisticActor, ActorValueOperator, ConvNet
from torchrl.modules.distributions.discrete import OneHotCategorical
from torch import nn
from torchrl.objectives.ppo import ClipPPOLoss
from collections import OrderedDict
from torchrl.objectives.value import GAE

def create_policy_loss(env, device=None):
    num_actions=env.action_spec.shape[-1]
    sample_obs=env.fake_tensordict()
    cnn_kwargs = {
        "num_cells": [32, 32, 32, 32],
        "kernel_sizes": [3, 3, 3,3],
        "strides": [2, 2, 2,2],
        "activation_class": nn.ELU,
        "paddings":[1,1,1,1]
    }
    av_common=TensorDictModule(
        module=nn.Sequential(OrderedDict(
                acc_conv=ConvNet(**cnn_kwargs, device=device),
                acc_linear=nn.LazyLinear(512, device=device),
    )),
        in_keys=["observation"],
        out_keys=["hidden"]
    )
    policy_module = ProbabilisticActor(
        module=TensorDictModule(
            module=nn.LazyLinear(out_features=num_actions, device=device),
            in_keys=["hidden"],
            out_keys=["logits"]
        ),
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )
    value_module = ValueOperator(
        module=nn.LazyLinear(out_features=1, device=device),
        in_keys=["hidden"],
        out_keys=["state_value"],
    )
    av_op=ActorValueOperator(av_common, policy_module, value_module)
    # init lazy
    av_op(sample_obs)
    
    policy_op=av_op.get_policy_operator()
    value_op=av_op.get_value_operator()
    loss_module = ClipPPOLoss(
        actor=policy_op,
        critic=value_op,
        value_key="state_value",
    )
    adv_module=GAE(gamma=0.99, lmbda=0.95, value_network=value_op,value_key="state_value")
    
    return policy_op, loss_module, adv_module

def info_dict_reader(info, td):
    for key in ["flag_get","time","x_pos","y_pos",]:
        if key in info: td.set(key,info[key])

from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from gymnasium.wrappers import RecordVideo  # bugs in RecordVideo within gym 0.26 
from gymnasium.experimental.wrappers.stateful_observation import MaxAndSkipObservationV0
def create_smb_env(stage=1,world=1, *,device=None,seed=None, video_kwargs=None, truncstillx=None, **gym_kwargs):
    if video_kwargs and video_kwargs["video_folder"]:
        gym_kwargs.setdefault("render_mode","rgb_array")
    env = gym_super_mario_bros.make(f"SuperMarioBros-{stage}-{world}-v0", apply_api_compatibility=True, **gym_kwargs) #
    env=SmbResetInfo(env)
    env=FrameCopy(env)
    if video_kwargs and video_kwargs["video_folder"]:
        env=RecordVideo(env, **video_kwargs)
    env=JoypadSpace(env, SIMPLE_MOVEMENT)
    env=ReleaseFallingA(env)
    env=ResizeObservation(env, 84)
    env=GrayScaleObservation(env, keep_dim=False)
    env=MaxAndSkipObservationV0(env, 4)
    if truncstillx is not None:
        env=TruncStillX(env, truncstillx)   # mainly for eval, wrap after skip, parameter for steps of policy (rather than frames behind policy)
    env=SmbReward(env,)
    env=FrameStack(env, 4)
    assert env.reset()[0].shape==(4, 84,84)  # .observation_space.shape
    env=GymWrapper(env, )
    env.info_dict_reader=info_dict_reader
    # for ParallelEnv, and for step_mdp to auto move keys from next to root
    env.observation_spec["observation"]=env.observation_spec["observation"].to(torch.float)     # image pixels
    env.observation_spec["flag_get"]=BinaryDiscreteTensorSpec(1,dtype=torch.bool)
    env.observation_spec["x_pos"]=UnboundedContinuousTensorSpec((1,),dtype=torch.float16)
    env.observation_spec["y_pos"]=UnboundedContinuousTensorSpec((1,),dtype=torch.float16)
    env.observation_spec["time"]=UnboundedContinuousTensorSpec((1,),dtype=torch.int16)

    env=TransformedEnv(env, 
                        Compose(
                            ObservationNorm(loc=0.0, scale=1/255.0, in_keys=["observation"]),
                            StepCounter(),
                            )
                        )
    if seed is not None:
        env.set_seed(seed=seed)     # cannot reproduce all episodes even with seed
    if device is not None:
        env=env.to(device)
    return env


base_dir=os.path.dirname(__file__)


def get_chkpt_file(run_tag): return os.path.join(base_dir, "chkpts", f"smb_actor-{run_tag}-flag.pt")

import re
def stage_world_from_run_tag(run_tag):
    sw=re.search(r"\b(s\d)\W?(w\d)\b",run_tag) or ["s1w1","s1","w1"]      # regex optim: ignore
    s, w=sw[1], sw[2]
    s=int(s[1:])
    w=int(w[1:])
    return s, w

default_seed=None
default_run_tag= None

def train(conf):
    
    
    run_tag=conf.run_tag or default_run_tag
    seed=conf.seed or default_seed
    tb_logdir=os.path.join(base_dir, "tb_logs","smb", run_tag)
    train_state_chkpt_file_flag=os.path.join(base_dir, "chkpts", f"smb-train-state-{run_tag}-flag.pt")
    train_state_chkpt_file_final=os.path.join(base_dir, "chkpts", f"smb-train-state-{run_tag}-final.pt")
    train_state_chkpt_file_bestx=os.path.join(base_dir, "chkpts", f"smb-train-state-{run_tag}-bestx.pt")
    chkpt_file=get_chkpt_file(run_tag)
    train_from_file=conf.train_from
    pretrained_file=conf.pretrained
    os.makedirs(os.path.dirname(chkpt_file), exist_ok=True)
    device=conf.device or "cuda:0" if torch.cuda.is_available() else "cpu"
    num_collectors=1
    num_vec_env=8
    total_frames=5000000
    T=256
    frames_per_batch=T*num_collectors*num_vec_env
    
    replay_buffer_capacity=2*num_vec_env*num_collectors
    replay_batch_size=8
    num_replay_sample=8*replay_buffer_capacity//replay_batch_size   # mostly * rounds
    max_grad_norm=0.5

    logger=setup_logger("train")
    
    stage, world= stage_world_from_run_tag(run_tag)
    logger.info(f"run_tag: {run_tag}, stage: {stage}, world: {world}")
    
    logger.info("Starting training")

    def create_serial_env():
        return create_smb_env(stage=stage, world=world, device=device,seed=seed)
        
    def create_vec_or_ser_env():
        if num_vec_env<=1:
            return create_serial_env()
        else:
            return ParallelEnv(num_vec_env, create_serial_env)
    
    env=create_serial_env()
    policy, loss_module, adv_module = create_policy_loss(env, device=device)
    
    writer=SummaryWriter(tb_logdir)
    writer.add_graph(policy, env.fake_tensordict()["observation"])
    
    # del env
    
    # OOM risk (32GB GPU mem) if gpu storage
    def create_replay_buffer(replay_buffer_capacity):
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.8, beta=0.9,
            storage=LazyTensorStorage(replay_buffer_capacity),  # store in cpu
            batch_size=replay_batch_size,
        )
        # replay_buffer=TensorDictReplayBuffer(
        #     storage=LazyTensorStorage(replay_buffer_capacity),
        #     sampler=SamplerWithoutReplacement(),
        #     batch_size=replay_batch_size,
        # )
        return replay_buffer

    replay_buffer=create_replay_buffer(replay_buffer_capacity)
    
    model=loss_module
    optimizer=torch.optim.Adam(model.parameters(), lr=1E-3)
    lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_frames//frames_per_batch*num_replay_sample)
    
    def create_collector():
        if num_collectors>1:
            return MultiSyncDataCollector(
                [create_vec_or_ser_env]*num_collectors,
                policy=policy,
                total_frames=total_frames,
                frames_per_batch=frames_per_batch,
                # device=device,  # with same device to of policy
                # split_trajs=True,
            )
        else:
            return SyncDataCollector(
                create_vec_or_ser_env,
                policy=policy,
                total_frames=total_frames,
                frames_per_batch=frames_per_batch,
                # device=device,  # with same device to of policy
                # split_trajs=True,
            )
    
    collector=create_collector()
    
    optim_steps=0
    trained_max_x_pos=0
    best_x_pos=0
    
    def save_train_state(sig=None,frame=None, file=train_state_chkpt_file_final):
        to_save={
            "policy": policy.state_dict(),
            "optim": optimizer.state_dict(),
            "optim_steps": optim_steps,
            "trained_max_x_pos": trained_max_x_pos,
            "best_x_pos": best_x_pos,
            # "collector": collector.state_dict(),      # error with par env
            # "replay_buffer":replay_buffer.state_dict(),
            "loss": loss_module.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        torch.save(to_save, file)
        logger.info(f"training state saved, with seen max x_pos: {trained_max_x_pos}")
        if sig is not None:
            raise SystemExit(sig)
    import signal
    signal.signal(signal.SIGTERM, save_train_state)
    signal.signal(signal.SIGINT, save_train_state)

    # load train state
    if train_from_file:
        state_map=torch.load(train_from_file, map_location=device)
        loss_module.load_state_dict(state_map["loss"])     # policy and value net weights
        optimizer.load_state_dict(state_map["optim"])
        best_x_pos=state_map["best_x_pos"]
        # trained_max_x_pos=state_map["trained_max_x_pos"]  # no
        lr_scheduler.load_state_dict(state_map["lr_scheduler"])
        
        logger.info(f"loaded train state from {train_from_file}")
        
    if pretrained_file:
        state_map=torch.load(pretrained_file, map_location=device)
        loss_module.load_state_dict(state_map["loss"])     # policy and value net weights
        logger.info(f"loaded pretrained actor & value weights from {pretrained_file}")

    @torch.no_grad()
    def eval_train_sync(i_eval_step, n_frame):
        nonlocal best_x_pos
        prev_training=policy.training
        policy.eval()
        etd=eval_train(policy, env)
        policy.train(prev_training)
        total_reward=etd["total_reward"]
        x_pos=etd["next","x_pos"]
        if x_pos>best_x_pos:
            best_x_pos=x_pos
            save_train_state(file=train_state_chkpt_file_bestx)
            logger.info(f"reaching further x_pos {best_x_pos} at frame {n_frame}")
        if etd["next","flag_get"]:
            torch.save(policy.state_dict(), chkpt_file)
            save_train_state(file=train_state_chkpt_file_flag)
            logger.info(f"FLAG GOT at frame {n_frame}")
            
        if i_eval_step is not None:
            writer.add_scalar("eval/max_step", etd["next","step_count"], i_eval_step)
            writer.add_scalar("eval/flag_get", etd["next","flag_get"], i_eval_step)
            writer.add_scalar("eval/x_pos", etd["next","x_pos"], i_eval_step)
            writer.add_scalar("eval/total_reward", total_reward , i_eval_step)
            writer.add_scalar("eval/action_efficiency", etd["next","x_pos"].to(torch.float)/(etd["next","step_count"]), i_eval_step)

    def save_policy_state(sig=None, frame=None):
        eval_train_sync(None, -1)
        logger.info(f"seen max x_pos: {trained_max_x_pos}")
    signal.signal(signal.SIGUSR1, save_policy_state)

    
    eval_batch_every=2
    with tqdm(total=total_frames, desc="Frames") as pbar:
        for i_batch, td in enumerate(collector):
            if isinstance(collector, SyncDataCollector) and isinstance(adv_module,GAE) and td.batch_dims==1:
                td=td.unsqueeze(0)      # => [T, F...] => [1,T, F...]
            if isinstance(adv_module, GAE) and td.batch_dims<=1:
                raise ValueError("unexpected batch size for GAE input")
            
            adv_module(td)
            replay_buffer.extend(td.cpu())
            for i_replay in range(num_replay_sample):
                sampled_td=replay_buffer.sample().to(device)
                loss_input_td=sampled_td
                
                loss_td=loss_module(loss_input_td)
                
                loss=sum([loss_td[key] for key in loss_td.keys() if key.startswith("loss_")])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                
                # loss_input_td["priority"]=torch.log(loss_input_td["x_pos"])     # loss could be negative
                replay_buffer.update_tensordict_priority(loss_input_td)
                
                trained_x_pos=loss_input_td["x_pos"].max().item()
                if trained_x_pos> trained_max_x_pos:
                    trained_max_x_pos=trained_x_pos
                
                writer.add_scalar("train/loss", loss, optim_steps)
                writer.add_scalar("train/reward", loss_input_td["next","reward"].mean(), optim_steps)
                for name, p in model.named_parameters():
                    writer.add_scalar(f"grads_norm/{name}", torch.norm(p.grad), optim_steps)
                    writer.add_scalar(f"weights_norm/{name}", torch.norm(p.data),  optim_steps)
                for key in filter(lambda x: x.startswith("loss_"), loss_td.keys()):
                    writer.add_scalar(f"train/{key}", loss_td[key], optim_steps)
                
                optim_steps+=1

            pbar.update(td.numel())
            
            if i_batch%eval_batch_every==0:
                i_eval_step=i_batch//eval_batch_every
                eval_train_sync(i_eval_step, pbar.n)
                
    
    if eval_batch_every>1:
        eval_train_sync(i_eval_step, pbar.n)

    logger.info("train completed")

    # save training state
    save_train_state(file=train_state_chkpt_file_final)
    logger.info("all done")
    del collector, replay_buffer, env

def maybe_moving(td):
    x_pos=td["x_pos"].to(torch.float)
    y_pos=td["y_pos"].to(torch.float)
    # time=td["next","time"]
    return not td["next","done"].any() and (td.numel()==1 or x_pos.std()>1E-3 and y_pos.std()>1E-3)

# def eval_train_async(pipe_parent, pipe_child, train_policy, tb_logdir, chkpt_file):
#     # pipe_parent.close()
#     writer=SummaryWriter(tb_logdir)
#     policy_device = next(policy.parameters()).device
#     env=create_smb_env(policy_device,seed=seed)
#     best_x_pos=100
#     pipe_child.send("inited")
#     policy=create_policy()
#
#     timeout=1   # sec
#     while True:
#         if pipe_child.poll(timeout):
#             msg,i_eval_step,n_frame =pipe_child.recv()
#             if i_eval_step is None:     # log if from SIGUSR1
#                 pipe_child.send("got")
#         else:
#             continue
#         if msg=="eval":
#             
#             policy.load_state_dict(train_policy.state_dict())
#             etd=eval_train(policy, env)
#             total_reward=etd["total_reward"]
#             if i_eval_step is not None:
#                 writer.add_scalar("eval/max_step", etd["next","step_count"], i_eval_step)
#                 writer.add_scalar("eval/flag_get", etd["next","flag_get"], i_eval_step)
#                 writer.add_scalar("eval/x_pos", etd["next","x_pos"], i_eval_step)
#                 writer.add_scalar("eval/total_reward", total_reward , i_eval_step)
#             x_pos=etd["x_pos"]
#             if x_pos>best_x_pos:
#                 torch.save(policy.state_dict(), chkpt_file)
#                 best_x_pos=x_pos
#                 print(f"model saved at frame {n_frame}, with best x_pos {best_x_pos}")
#         elif msg=="close":
#             writer.close()
#             env.close()
#             break
#         else:
#             raise RuntimeError("unknown command")

@torch.no_grad()
def eval_train(policy, env, eval_batch_size=32, loss_module=None, max_eval_steps=1000):
    last_step=env.reset()
    losses=[]
    total_reward=0.0
    eval_batch_size=min(eval_batch_size,max_eval_steps)
    while True:
        steps=env.rollout(eval_batch_size,policy, auto_reset=False,tensordict=last_step)
        total_reward+=steps["next","reward"].sum()
        last_step=steps[-1]
        if loss_module is not None:
            loss_td=loss_module(steps)
            losses.append(loss_td)
        if not maybe_moving(steps) or last_step["next","step_count"]>=max_eval_steps: break
    last_step["total_reward"]=total_reward
    if loss_module is not None:
            last_step["loss"]=torch.stack(losses).mean()
    return last_step

def policy_state(state_map):
    """
    Loading from policy state chkpt or train state chkpt.
    """
    if "policy" in state_map and isinstance(state_map["policy"], dict):
        state_map=state_map["policy"]
    return state_map

@torch.no_grad()
def eval_video(conf):
    logger=setup_logger("eval-video")
    logger.info("Starting evaluation")
    device=conf.device or "cuda:0" if torch.cuda.is_available() else "cpu"
    run_tag=conf.run_tag or default_run_tag
    stage,world=stage_world_from_run_tag(run_tag)
    eval_chkpt_file=conf.chkpt_file or get_chkpt_file(run_tag or eval_chkpt_file)
    env=create_smb_env(stage=stage, world=world, device=device, seed=conf.seed or default_seed, 
                       video_kwargs=dict(video_folder=os.path.join(base_dir,"video"), name_prefix=f"smb-{run_tag}"),
                       truncstillx=64)
    policy, *_ = create_policy_loss(env,device)
    policy.load_state_dict(policy_state(torch.load(eval_chkpt_file, map_location=device)))
    logger.info("loaded chkpt: %s", eval_chkpt_file)
    td=env.rollout(1000, policy)
    last_step=td[-1]
    logger.info("max step: %s", last_step["next","step_count"].item())
    logger.info("flag_get: %s", last_step["next","flag_get"].item())
    logger.info("max_x: %s", last_step["next","x_pos"].item())
    env.close()
    # del env # trigger video saving

@torch.no_grad()
def eval_render(conf):
    logger=setup_logger("eval-render")
    logger.info("evaluating...")
    device=conf.device or "cuda:0" if torch.cuda.is_available() else "cpu"
    run_tag=conf.run_tag or default_run_tag
    eval_chkpt_file=conf.chkpt_file or get_chkpt_file(run_tag)
    stage,world=stage_world_from_run_tag(run_tag or eval_chkpt_file)
    env=create_smb_env(stage=stage, world=world, device=device, seed=conf.seed or default_seed, truncstillx=64, render_mode="human")
    policy, *_ = create_policy_loss(env,device)
    policy.load_state_dict(policy_state(torch.load(eval_chkpt_file, map_location=device)))
    logger.info("loaded chkpt: %s", eval_chkpt_file)
    td=env.rollout(1000, policy)
    last_step=td[-1]
    logger.info("max step: %s", last_step["next","step_count"].item())
    logger.info("flag_get: %s", last_step["next","flag_get"].item())
    logger.info("max_x: %s", last_step["next","x_pos"].item())
    del env

import argparse
if __name__ == '__main__':
    arg_parser=argparse.ArgumentParser(__file__)
    arg_parser.add_argument("--eval_render", action="store_true",)
    arg_parser.add_argument("--eval_video", action="store_true",)
    arg_parser.add_argument("--device", action="store",)
    arg_parser.add_argument("--chkpt", action="store", dest="chkpt_file")
    arg_parser.add_argument("--run_tag", action="store",)
    arg_parser.add_argument("--train_from", action="store")
    arg_parser.add_argument("--pretrained", action="store")
    arg_parser.add_argument("--seed", type=int, action="store")
    args=arg_parser.parse_args()
    
    if args.eval_render:
        eval_render(args)
    elif args.eval_video:
        eval_video(args)
    else:
        train(args)
