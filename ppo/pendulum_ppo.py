import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from torchrl.collectors import MultiSyncDataCollector,SyncDataCollector
from torchrl.data import TensorSpec, UnboundedContinuousTensorSpec,BinaryDiscreteTensorSpec
from torchrl.envs.libs.gym import GymEnv
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchrl.data.replay_buffers import TensorDictPrioritizedReplayBuffer,ReplayBuffer, ListStorage,LazyTensorStorage,LazyMemmapStorage,SamplerWithoutReplacement,RandomSampler,PrioritizedSampler,TensorDictReplayBuffer
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import TransformedEnv, Compose, ToTensorImage,StepCounter,\
    R3MTransform,ObservationNorm, ParallelEnv, FrameSkipTransform,RenameTransform,ExcludeTransform,set_exploration_mode
import os
from ignite.utils import setup_logger
from torchrl.envs.utils import step_mdp
from torch import multiprocessing as mp
from torchrl.envs.transforms import Transform, Compose, ObservationNorm,DoubleToFloat, RewardScaling
from torchrl.envs.utils import check_env_specs
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule,NormalParamExtractor
from torchrl.modules import ValueOperator, ProbabilisticActor, ActorValueOperator, MLP, ConvNet,TanhNormal
from torchrl.modules.distributions.discrete import OneHotCategorical
from torchrl.objectives.value import GAE
from torch import nn
from torchrl.objectives.ppo import ClipPPOLoss
from collections import OrderedDict
from torchrl.objectives.utils import ValueEstimators

def create_policy_loss(env, *, hidden_dim=512, device=None):
    num_cells = hidden_dim
    num_actions=env.action_spec.shape[-1]
    sample_obs=env.fake_tensordict().to(device)
    
    av_commone_net=nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
    )
    av_common_module=TensorDictModule(
        module=av_commone_net,
        in_keys=["observation"],
        out_keys=["hidden"]
    )
    policy_module = ProbabilisticActor(
        module=TensorDictModule(
            module=nn.Sequential(
                nn.LazyLinear(2 * num_actions, device=device),
                NormalParamExtractor(),),
            in_keys=["hidden"],
            out_keys=["loc", "scale"]
        ),
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.action_spec.space.minimum,
            "max": env.action_spec.space.maximum,
        },
        return_log_prob=True,
    )
    value_module = ValueOperator(
        module=nn.LazyLinear(out_features=1, device=device),
        in_keys=["hidden"],
        out_keys=["state_value"],
    )

    av_operator = ActorValueOperator(av_common_module, policy_module, value_module)
    # init lazy
    av_operator(sample_obs)
    
    policy_op=av_operator.get_policy_operator()
    value_op=av_operator.get_value_operator()
    loss_module = ClipPPOLoss(
        actor=policy_op,
        critic=value_op,
        value_key="state_value",
    )
    adv_module=GAE(gamma=0.9, lmbda=0.95,value_network=value_op, value_key="state_value")
    
    return policy_op, loss_module, adv_module

def create_env(device=None,seed=None, video_kwargs=None, **gym_kwargs):
    if video_kwargs and video_kwargs["video_folder"]:
        gym_kwargs.setdefault("render_mode","rgb_array")
    env=gym.make("Pendulum-v1", **gym_kwargs)
    if video_kwargs and video_kwargs["video_folder"]:
        env=gym.wrappers.RecordVideo(env, **video_kwargs)
    env=GymWrapper(env, )
    if device is not None:
        env=env.to(device)
    env=TransformedEnv(env, 
                        Compose(
                            # ObservationNorm(in_keys=["observation"]),
                            DoubleToFloat(in_keys=["observation"]),
                            RewardScaling(loc=0.0, scale=1/16.0),
                            StepCounter(),
                            )
                        )
    if seed is not None:
        env.set_seed(seed=seed)
    # env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    # check_env_specs(env)
    return env


base_dir=os.path.dirname(__file__)
run_tag="exp0"
seed=None

def get_chkpt_file(run_tag): return os.path.join(base_dir, "chkpts", f"pendulum-{run_tag}.pt")


def train(conf):
    
    

    tb_logdir=os.path.join(base_dir, "tb_logs","pendulum", run_tag)
    train_state_chkpt_file_best=os.path.join(base_dir, "chkpts", f"pendulum-train-state-{run_tag}.pt")
    train_state_chkpt_file_final=os.path.join(base_dir, "chkpts", f"pendulum-train-state-{run_tag}-final.pt")
    chkpt_file=get_chkpt_file(run_tag)
    train_from_file=conf.train_from
    os.makedirs(os.path.dirname(chkpt_file), exist_ok=True)
    device=conf.device or "cuda:0" if torch.cuda.is_available() else "cpu"
    num_collectors=1
    num_vec_env=8
    total_frames=1000000
    T=64
    frames_per_batch=T*num_collectors*num_vec_env
    
    replay_buffer_capacity=2*num_vec_env*num_collectors
    replay_batch_size=1
    num_replay_sample=10*replay_buffer_capacity//replay_batch_size   # mostly * rounds
    max_grad_norm=0.5

    logger=setup_logger("train")
    
    logger.info("Starting training")

    def create_serial_env():
        return create_env(device=device,seed=seed)
        
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
        # replay_buffer = TensorDictPrioritizedReplayBuffer(
        #     alpha=0.7, beta=0.9,
        #     storage=LazyTensorStorage(replay_buffer_capacity),  # store in cpu
        #     batch_size=replay_batch_size
        # )
        replay_buffer=TensorDictReplayBuffer(
            storage=LazyTensorStorage(replay_buffer_capacity),
            sampler=SamplerWithoutReplacement(),
            batch_size=replay_batch_size,
        )
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
    seen_best_score=0
    best_score=-1000
    
    def save_train_state(sig=None,frame=None, file=train_state_chkpt_file_final):
        to_save={
            "policy": policy.state_dict(),
            "optim": optimizer.state_dict(),
            "optim_steps": optim_steps,
            "seen_best_score": seen_best_score,
            "best_score": best_score,
            # "collector": collector.state_dict(),      # error with par env
            # "replay_buffer":replay_buffer.state_dict(),
            "loss": loss_module.state_dict(),
            "lr_scheduler":  lr_scheduler.state_dict(),
        }
        torch.save(to_save, file)
        logger.info(f"training state saved, with seen max step: {seen_best_score}")
        if sig is not None:
            raise SystemExit(sig)
    import signal
    signal.signal(signal.SIGTERM, save_train_state)
    signal.signal(signal.SIGINT, save_train_state)

    # load train state
    if train_from_file:
        state_map=torch.load(train_from_file, map_location=device)
        loss_module.load_state_dict(state_map["loss"])     # including policy parameters
        optimizer.load_state_dict(state_map["optim"])
        best_score=state_map["best_score"]
        # seen_best_score=state_map["seen_best_score"]  # no
        lr_scheduler.load_state_dict(state_map["lr_scheduler"])
        
        logger.info(f"loaded train state from {train_from_file}")

    @torch.no_grad()
    def eval_train_sync(i_eval_step, n_frame):
        nonlocal best_score
        prev_training=policy.training
        policy.eval()
        max_eval_steps=200
        with set_exploration_mode("mean"):
            eroll=env.rollout(max_eval_steps, policy)
        policy.train(prev_training)
        score=eroll["next","reward"].sum().item()
        if eroll["next","step_count"].max().item()==max_eval_steps and score==best_score:
            torch.save(policy.state_dict(), os.path.join(base_dir, "chkpts", f"pendulum-policy-{run_tag}-peak-final.pt"))
            logger.info(f"max eval steps reached at frame {n_frame}")
        
        if i_eval_step is not None:
            writer.add_scalar("eval/max_step", eroll["next","step_count"].max().item(), i_eval_step)
            writer.add_scalar("eval/reward", eroll["next","reward"].mean().item(), i_eval_step)
            writer.add_scalar("eval/total_reward", eroll["next","reward"].sum().item() , i_eval_step)
        if score>best_score:
            best_score=score
            torch.save(policy.state_dict(), chkpt_file)
            logger.info(f"model saved at frame {n_frame}, with best max_step {best_score}")
            save_train_state(file=train_state_chkpt_file_best)

    def save_policy_state(sig, frame=None):
        eval_train_sync(None, -1)
        logger.info(f"seen max step: {seen_best_score}")
    signal.signal(signal.SIGUSR1, save_policy_state)
    
    eval_batch_every=1
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
                
                seen_score=loss_input_td["next","step_count"].max().item()
                if seen_score> seen_best_score:
                    seen_best_score=seen_score
                
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
    del collector, replay_buffer


@torch.no_grad()
def eval_video(conf):
    logger=setup_logger("eval-video")
    logger.info("Starting evaluation")
    device=conf.device or "cuda:0" if torch.cuda.is_available() else "cpu"
    eval_run_tag=conf.run_tag or run_tag
    eval_seed=conf.seed or seed
    env=create_env()
    policy, *_ = create_policy_loss(env,device=device)
    env=create_env(device=device, seed=eval_seed, video_kwargs=dict(video_folder=os.path.join(base_dir,"video"), name_prefix=f"pendulum-{eval_run_tag}"))
    eval_chkpt_file=conf.chkpt_file or get_chkpt_file(eval_run_tag)
    policy.load_state_dict(torch.load(eval_chkpt_file, map_location=device))
    logger.info("loaded chkpt: %s", eval_chkpt_file)
    policy.eval()
    with set_exploration_mode("mean"):
        td=env.rollout(200, policy)
    logger.info("total reward: %s", td["next", "reward"].sum().item())
    env.close() # trigger video saving

@torch.no_grad()
def eval_render(conf):
    logger=setup_logger("eval-render")
    logger.info("evaluating...")
    device=conf.device or "cpu"
    eval_run_tag=conf.run_tag or run_tag
    env=create_env(device=device, seed=seed, render_mode="human")
    policy, *_ = create_policy_loss(env,device=device)
    eval_chkpt_file=conf.chkpt_file or get_chkpt_file(eval_run_tag)
    policy.load_state_dict(torch.load(eval_chkpt_file, map_location=device))
    policy.eval()
    logger.info("loaded chkpt: %s", eval_chkpt_file)
    with set_exploration_mode("mean"):
        td=env.rollout(200, policy)
    logger.info("total reward: %s", td["next", "reward"].sum().item())
    del env

import argparse
if __name__ == '__main__':
    arg_parser=argparse.ArgumentParser(__file__)
    arg_parser.add_argument("--eval_render", action="store_true",)
    arg_parser.add_argument("--eval_video", action="store_true",)
    arg_parser.add_argument("--device", action="store",)
    arg_parser.add_argument("--chkpt_file", action="store",)
    arg_parser.add_argument("--run_tag", action="store",)
    arg_parser.add_argument("--train_from", action="store")
    arg_parser.add_argument("--seed", type=int, action="store")
    args=arg_parser.parse_args()
    
    if args.eval_render:
        eval_render(args)
    elif args.eval_video:
        eval_video(args)
    else:
        train(args)
    
