import wandb, os
import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim
from class_snapbot import Snapbot4EnvClass
from utils import *
from class_pid import PIDControllerClass
from class_grp import GaussianRandomPathClass, scaleup_traj, get_anchors_from_traj
from class_dlpg import DeepLatentPolicyGradientClass

class SnapbotTrajectoryUpdateClass():
    def __init__(self,
                name = "SnapbotTrajectoryUpdateClass",
                env  = None,
                k_p  = 0.2,
                k_i  = 0.001,
                k_d  = 0.01,
                out_min = -2,
                out_max = +2, 
                ANTIWU  = True,
                z_dim    = 32,
                c_dim    = 3,
                h_dims   = [128, 128],
                var_max  = 0.1,
                n_anchor = 20,
                dur_sec  = 2,
                max_repeat    = 5,
                hyp_prior     = {'g': 1/1, 'l': 1/8, 'w': 1e-8},
                hyp_posterior = {'g': 1/4, 'l': 1/8, 'w': 1e-8},
                lbtw_base     = 0.8,
                device_idx = 0,
                VERBOSE    = True
                ):
        # Init params
        self.name       = name
        self.env        = env
        self.z_dim      = z_dim
        self.c_dim      = c_dim
        self.n_anchor   = n_anchor
        self.dur_sec    = dur_sec   
        self.max_repeat = max_repeat
        self.hyp_prior      = hyp_prior
        self.hyp_poseterior = hyp_posterior
        self.lbtw_base   = lbtw_base
        try: 
            self.device  = torch.device('mps')
        except:
            self.device  = torch.device('cuda:{}'.format(device_idx) if torch.cuda.is_available() else 'cpu')
        self.VERBOSE     = VERBOSE
        # Set grp & pid & qscaler
        self.PID  = PIDControllerClass(name="PID", k_p=k_p, k_i=k_i, k_d=k_d, dim=self.env.adim, out_min=out_min, out_max=out_max, ANTIWU=ANTIWU)
        self.DLPG = DeepLatentPolicyGradientClass(name='DLPG', x_dim=env.adim*n_anchor, c_dim=c_dim, z_dim=z_dim, h_dims=h_dims, actv_enc=nn.ReLU(), actv_dec=nn.ReLU(), actv_q=nn.Softplus(), actv_out=None, var_max=var_max, device=self.device)
        self.QScaler      = ScalerClass(obs_dim=1)
        self.GRPPrior     = GaussianRandomPathClass(name='GRP Prior')
        self.GRPPosterior = GaussianRandomPathClass(name='GRP Posterior')
        # Set model to device
        self.DLPG.to(self.device)
        # Set explaination
        if self.VERBOSE:
            print("{} START with DEVICE: {}".format(self.name, self.device))

    def update(self,
                seed = 0,
                start_epoch = 0,
                max_epoch   = 500,
                n_sim_roll          = 100,
                sim_update_size     = 64,
                n_sim_update        = 64,
                n_sim_prev_consider = 10,
                n_sim_prev_best_q   = 50,
                init_prior_prob = 0.5,
                folder = 0,
                WANDB  = False
                ):
        # Set random seed 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Set wandb 
        if WANDB:
            wandb.init(project="snapbot_sim", entity="hansooworld")

        # Set buffer
        sim_x_list  = np.zeros((n_sim_roll, self.env.adim*self.n_anchor))
        sim_c_list  = np.zeros((n_sim_roll, self.c_dim))
        sim_q_list  = np.zeros(n_sim_roll)
        sim_x_lists = [''] * max_epoch
        sim_c_lists = [''] * max_epoch
        sim_q_lists = [''] * max_epoch

        while start_epoch < max_epoch:
            train_rate        = start_epoch / max_epoch
            exp_decrease_rate = np.exp(-5 * (train_rate**2)) # 1 -> 0
            exp_increase_rate = 1 - exp_decrease_rate        # 0 -> 1
            prior_prob = init_prior_prob * exp_decrease_rate # Schedule eps-greedish (init_prior_prob -> 0)
            lbtw       = self.lbtw_base + (1-self.lbtw_base)*exp_increase_rate # Schedule leveraged GRP (0.8 -> 1.0)
            lbtw       = lbtw * 0.9 # max leverage to be 0.9

            for sim_idx in range(n_sim_roll):
                exploration_coin = np.random.rand()
                if self.env.condition is not None:
                    condition_coin = np.random.uniform(0, 1)
                    if condition_coin >= 0.7:
                        c = np.array([1,0,0])
                    elif condition_coin >= 0.3:
                        c = np.array([0,1,0])
                    else:
                        c = np.array([0,0,1])
                else:
                    c = np.array([0,1,0])
                if (exploration_coin < prior_prob) or (start_epoch < 1):
                    self.GRPPrior.set_prior(n_data_prior=4, dim=self.env.adim, dur_sec=self.dur_sec, HZ=self.env.hz, hyp=self.hyp_prior)
                    traj_joints, traj_secs = self.GRPPrior.sample_one_traj(rand_type='Uniform', ORG_PERTURB=True, perturb_gain=0.0) 
                    traj_joints_deg = scaleup_traj(self.env, traj_joints, DO_SQUASH=True, squash_margin=5)
                else:
                    x_anchor = self.DLPG.sample_x(c=torch.FloatTensor(c).reshape(1,-1).to(self.device), n_sample=1, SKIP_Z_SAMPLE=True).reshape(self.n_anchor, self.env.adim)
                    x_anchor = x_anchor.detach().cpu().numpy()
                    x_anchor[-1,:] = x_anchor[0,:]
                    self.GRPPosterior.set_posterior(t_anchor, x_anchor,lbtw=lbtw, t_test=traj_secs, hyp=self.hyp_poseterior, APPLY_EPSRU=True, t_eps=0.025)
                    traj_joints, _ = self.GRPPosterior.sample_one_traj(rand_type='Uniform', ORG_PERTURB=True, perturb_gain=0.0) 
                    traj_joints_deg = scaleup_traj(self.env, np.array(traj_joints), DO_SQUASH=True, squash_margin=5)
                t_anchor, x_anchor = get_anchors_from_traj(traj_secs, traj_joints, n_anchor=self.n_anchor) 
                result = rollout(self.env, self.PID, traj_joints_deg, n_traj_repeat=self.max_repeat)
                sim_x_list[sim_idx, :] = x_anchor.reshape(1,-1)
                sim_c_list[sim_idx, :] = c
                sim_q_list[sim_idx]    = sum(result['forward_rewards'])
            
            sim_x_lists[start_epoch] = sim_x_list
            sim_c_lists[start_epoch] = sim_c_list
            sim_q_lists[start_epoch] = sim_q_list

            for n_prev_idx in range(n_sim_prev_consider):
                if n_prev_idx == 0:
                    sim_x_list_bundle = sim_x_list
                    sim_c_list_bundle = sim_c_list
                    sim_q_list_bundle = sim_q_list
                else:
                    sim_x_list_bundle = np.concatenate((sim_x_list_bundle, sim_x_lists[max(0, start_epoch-n_prev_idx)]), axis=0)
                    sim_c_list_bundle = np.concatenate((sim_c_list_bundle, sim_c_lists[max(0, start_epoch-n_prev_idx)]), axis=0)
                    sim_q_list_bundle = np.concatenate((sim_q_list_bundle, sim_q_lists[max(0, start_epoch-n_prev_idx)]))
            
            sorted_idx  = np.argsort(-sim_q_list_bundle)
            sim_x_train = sim_x_list_bundle[sorted_idx[:n_sim_prev_best_q], :]
            sim_c_train = sim_c_list_bundle[sorted_idx[:n_sim_prev_best_q], :]
            sim_q_train = sim_q_list_bundle[sorted_idx[:n_sim_prev_best_q]]

            sim_x_train = np.concatenate((sim_x_train, sim_x_list), axis=0)
            sim_c_train = np.concatenate((sim_c_train, sim_c_list), axis=0)
            sim_q_train = np.concatenate((sim_q_train, sim_q_list))

            rand_idx    = np.random.permutation(sim_x_list_bundle.shape[0])[:n_sim_roll]
            sim_x_rand  = sim_x_list_bundle[rand_idx, :]
            sim_c_rand  = sim_c_list_bundle[rand_idx, :]
            sim_q_rand  = sim_q_list_bundle[rand_idx]
            sim_x_train = np.concatenate((sim_x_train, sim_x_rand), axis=0)
            sim_c_train = np.concatenate((sim_c_train, sim_c_rand), axis=0)
            sim_q_train = np.concatenate((sim_q_train, sim_q_rand))

            self.QScaler.reset()
            self.QScaler.update(sim_q_train)
            sim_q_scale, sim_q_offset = self.QScaler.get()
            sim_scaled_q = sim_q_scale * (sim_q_train-sim_q_offset)
            if start_epoch < 50:
                recon_loss_gain = 1
                beta = 0
            else:
                recon_loss_gain = 1
                beta = 1
            loss = self.DLPG.update(x=sim_x_train, c=sim_c_train, q=sim_scaled_q, lr=0.001, 
                                    recon_loss_gain=recon_loss_gain, beta=beta, max_iter=n_sim_update, batch_size=sim_update_size)

            # For eval
            x_anchor = self.DLPG.sample_x(c=torch.FloatTensor(c).reshape(1,-1).to(self.device), n_sample=1, SKIP_Z_SAMPLE=True).reshape(self.n_anchor, self.env.adim)
            x_anchor = x_anchor.detach().cpu().numpy()
            x_anchor[-1,:] = x_anchor[0,:]
            self.GRPPosterior.set_posterior(t_anchor, x_anchor, lbtw=0.9, t_test=traj_secs, hyp=self.hyp_poseterior, APPLY_EPSRU=True, t_eps=0.025)
            policy4eval_traj   = self.GRPPosterior.sample_one_traj(rand_type='Uniform', ORG_PERTURB=True, perturb_gain=0.0)[0]
            policy4eval_traj   = scaleup_traj(self.env, policy4eval_traj, DO_SQUASH=True, squash_margin=5)
            t_anchor, x_anchor = get_anchors_from_traj(traj_secs, policy4eval_traj, n_anchor=self.n_anchor) 
            policy4eval  = rollout(self.env, self.PID, policy4eval_traj, n_traj_repeat=self.max_repeat)
            eval_secs    = policy4eval['secs']
            eval_xy_degs = policy4eval['xy_degs']
            eval_reward  = sum(policy4eval['forward_rewards'])
            eval_x_diff  = policy4eval['x_diff']

            # For wandb
            if WANDB:
                wandb.log({"sim_reward": eval_reward, "sim_x_diff": eval_x_diff, "DLPG_loss": loss})

            # Save model weights
            if (start_epoch+1) % 50 == 0 and start_epoch != 0:
                if not os.path.exists("dlpg/{}/weights".format(folder)):
                    os.makedirs("dlpg/{}/weights".format(folder))
                torch.save(self.DLPG.state_dict(), 'dlpg/{}/weights/dlpg_model_weights_{}.pth'.format(folder, start_epoch+1))

            # For printing evaluation of present policy
            if (start_epoch+1) % 5 == 0 and start_epoch != 0:
                print("EPOCH: {:>3}, CONDITION: [{}], REWARD: {:>6.2f}, XDIFF: {:>4.3f}".format(start_epoch+1, c, eval_reward, eval_x_diff))
            
            # Save snapbot's trajectories
            if (start_epoch+1) % 20 == 0 and start_epoch != 0:
                if not os.path.exists("dlpg/{}/plot".format(folder)):
                    os.makedirs("dlpg/{}/plot".format(folder))
                plot_snapbot_joint_traj_and_topdown_traj(traj_secs, policy4eval_traj, t_anchor, x_anchor, eval_xy_degs, eval_secs,
                                                figsize=(16,8), title_str='EPOCH: {:>3} REWARD: {:>6.2f}'.format(start_epoch+1, eval_reward), 
                                                tfs=15, SAVE=True, image_name='dlpg/{}/plot/epoch_{}.png'.format(folder, start_epoch+1))
            print("[{:>3} / {}] Clear".format(start_epoch+1, max_epoch))
            start_epoch += 1

if __name__ == "__main__":
    env = Snapbot4EnvClass(render_mode=None)
    SnapbotTrajectoryUpdateClass = SnapbotTrajectoryUpdateClass(
                                                                name = "SnapbotTrajectoryUpdateClass",
                                                                env  = env,
                                                                k_p  = 0.2,
                                                                k_i  = 0.001,
                                                                k_d  = 0.01,
                                                                out_min = -2,
                                                                out_max = +2, 
                                                                ANTIWU  = True,
                                                                z_dim    = 32,
                                                                c_dim    = 3,
                                                                h_dims   = [128, 128],
                                                                var_max  = None,
                                                                n_anchor = 20,
                                                                dur_sec  = 2,
                                                                max_repeat    = 5,
                                                                hyp_prior     = {'g': 1/1, 'l': 1/8, 'w': 1e-8},
                                                                hyp_posterior = {'g': 1/4, 'l': 1/8, 'w': 1e-8},
                                                                lbtw_base     = 0.8,
                                                                device_idx = 1
                                                                )
    SnapbotTrajectoryUpdateClass.update(
                                        seed = 0,
                                        start_epoch = 0,
                                        max_epoch   = 300,
                                        n_sim_roll          = 100,
                                        sim_update_size     = 64,
                                        n_sim_update        = 64,
                                        n_sim_prev_consider = 10,
                                        n_sim_prev_best_q   = 50,
                                        init_prior_prob = 0.5,
                                        folder = 12,
                                        WANDB  = True
                                        )
