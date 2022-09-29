import torch, glob, os
import numpy as np
import matplotlib.image as mpimg
from class_snapbot import Snapbot4EnvClass, Snapbot3EnvClass
from class_policy_cvae import SnapbotTrajectoryUpdateClass
from class_grp import *
from utils import *

def eval_snapbot_from_network(env, dur_sec, n_anchor, max_repeat, folder, epoch,  condition, RENDER=False, PLOT=True):
    EvalPolicy = SnapbotTrajectoryUpdateClass(
                                                name = "EvalPolicy",
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
                                                n_anchor = n_anchor,
                                                dur_sec  = dur_sec,
                                                max_repeat    = max_repeat,
                                                hyp_prior     = {'g': 1/1, 'l': 1/8, 'w': 1e-8},
                                                hyp_posterior = {'g': 1/4, 'l': 1/8, 'w': 1e-8},
                                                lbtw_base     = 0.8,
                                                device_idx = 0
                                                )
    try:
        EvalPolicy.DLPG.load_state_dict(torch.load("dlpg/{}/weights/dlpg_model_weights_{}.pth".format(folder, epoch), map_location='cuda:0'))
    except:
        EvalPolicy.DLPG.load_state_dict(torch.load("dlpg/{}/weights/dlpg_model_weights_{}.pth".format(folder, epoch), map_location='cpu'))
    EvalPolicy.DLPG.eval()
    traj_joints, traj_secs = EvalPolicy.GRPPrior.sample_one_traj(rand_type='Uniform', ORG_PERTURB=True, perturb_gain=0.0)
    t_anchor, x_anchor = get_anchors_from_traj(traj_secs, traj_joints, n_anchor=EvalPolicy.n_anchor) 
    n_sample = 1
    for i in range(n_sample):
        x_anchor = EvalPolicy.DLPG.sample_x(c=torch.FloatTensor(condition).reshape(1,-1).to(EvalPolicy.device), n_sample=1, SKIP_Z_SAMPLE=True).reshape(EvalPolicy.n_anchor, EvalPolicy.env.adim)
        x_anchor = x_anchor.detach().cpu().numpy()
        x_anchor[-1,:] = x_anchor[0,:]
        EvalPolicy.GRPPosterior.set_posterior(t_anchor,x_anchor,lbtw=0.9,t_test=traj_secs,hyp=EvalPolicy.hyp_poseterior,APPLY_EPSRU=True,t_eps=0.025)
        # EvalPolicy.GRPPosterior.plot(n_sample=1,figsize=(15,5),subplot_rc=(2,4),lw_sample=1/2,tfs=10,rand_type='Uniform')
        policy4eval_traj, traj_secs = EvalPolicy.GRPPosterior.sample_one_traj(rand_type='Uniform', ORG_PERTURB=True, perturb_gain=0.0)
        policy4eval_traj = scaleup_traj(EvalPolicy.env, policy4eval_traj, DO_SQUASH=True, squash_margin=5)
        policy4eval = rollout(EvalPolicy.env, EvalPolicy.PID, policy4eval_traj, n_traj_repeat=EvalPolicy.max_repeat, RENDER=RENDER, PLOT=PLOT)
        eval_reward  = sum(policy4eval['forward_rewards'])
        eval_x_diff  = policy4eval['x_diff']
        eval_figure  = policy4eval['figure']
        eval_figure.savefig('for_plot_{}'.format(i))
        plt.close()
        print("REWARD: {:>.1f} X_DIFF: {:>.3f}".format(eval_reward, eval_x_diff))
    fig = plt.figure()
    rows = n_sample
    cols = 1
    i = 1
    for idx, filename in enumerate(glob.glob("*.png")):
        img = mpimg.imread(filename)
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(img)
        plt.axis('off')
        i += 1
        os.remove(filename)
        if idx == n_sample-1:
            break
    plt.show()

if  __name__ == "__main__":
    env = Snapbot3EnvClass(render_mode=None)
    eval_snapbot_from_network(env=env, dur_sec=2, n_anchor=20, max_repeat=5, folder=15, epoch=300,  condition=[0,1,0], RENDER=True, PLOT=True)