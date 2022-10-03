import argparse, json, string
from class_snapbot import Snapbot4EnvClass, Snapbot3EnvClass
from class_policy_vqvae import SnapbotTrajectoryUpdateClass

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        frac = float(num) / float(denom)
        return frac

def main(args):
    if args.env == int(4):
        env = Snapbot4EnvClass(render_mode=None)
    if args.env == int(3):
        env = Snapbot3EnvClass(render_mode=None)
    args.hyp_prior = json.loads(args.hyp_prior)
    args.hyp_posterior = json.loads(args.hyp_posterior)
    for k, v in args.hyp_prior.items():
        args.hyp_prior[k] = convert_to_float(v)
    for k, v in args.hyp_posterior.items():
        args.hyp_posterior[k] = convert_to_float(v)
    SnapbotTrajectoryUpdate = SnapbotTrajectoryUpdateClass(
                                                            name = "SnapbotTrajectoryUpdateClass",
                                                            env  = env,
                                                            k_p  = 0.2,
                                                            k_i  = 0.001,
                                                            k_d  = 0.01,
                                                            out_min = -args.torque,
                                                            out_max = +args.torque, 
                                                            ANTIWU  = True,
                                                            z_dim   = args.z_dim,
                                                            c_dim   = args.c_dim,
                                                            h_dims  = args.h_dims,
                                                            embedding_num   = args.embedding_num,
                                                            embedding_dim   = args.embedding_dim,
                                                            commitment_beta = args.commitment_beta,
                                                            n_anchor = args.n_anchor,
                                                            dur_sec  = args.dur_sec,
                                                            max_repeat    = args.max_repeat,
                                                            hyp_prior     = args.hyp_prior,  
                                                            hyp_posterior = args.hyp_posterior,
                                                            lbtw_base     = args.lbtw_base,
                                                            device_idx = args.device_idx, # -1 to use RAY
                                                            VERBOSE    = args.VERBOSE,
                                                            WANDB  = args.WANDB,
                                                            args   = args
                                                            )

    SnapbotTrajectoryUpdate.update(
                                    seed        = args.seed,
                                    lr_dlpg     = args.lr_dlpg,
                                    eps_dlpg    = args.eps_dlpg,
                                    n_worker    = args.n_worker,
                                    start_epoch = args.start_epoch,
                                    max_epoch   = args.max_epoch,
                                    n_sim_roll          = args.n_sim_roll,
                                    sim_update_size     = args.sim_update_size,
                                    n_sim_update        = args.n_sim_update,
                                    n_sim_prev_consider = args.n_sim_prev_consider,
                                    n_sim_prev_best_q   = args.n_sim_prev_best_q,
                                    init_prior_prob = args.init_prior_prob,
                                    folder = args.folder
                                    )

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=4, help="setting for env", type=int)
    parser.add_argument("--torque", default=2, help="setting for max torque", type=float)
    parser.add_argument("--z_dim", default=32, type=int)
    parser.add_argument("--c_dim", default=3, type=int)
    parser.add_argument("--h_dims", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--embedding_num", default=10, type=int)
    parser.add_argument("--embedding_dim", default=32, type=int)
    parser.add_argument("--commitment_beta", default=2, type=float)
    parser.add_argument("--n_anchor", default=20, type=int)
    parser.add_argument("--dur_sec", default=2, type=float)
    parser.add_argument("--max_repeat", default=5, type=int)
    parser.add_argument("--hyp_prior", default={'g': 1/1, 'l': 1/8, 'w': 1e-8}, type=str)
    parser.add_argument("--hyp_posterior", default={'g': 1/1, 'l': 1/8, 'w': 1e-8}, type=str)
    parser.add_argument("--lbtw_base", default=0.8, type=float)
    parser.add_argument("--device_idx", default=-1, type=int)
    parser.add_argument("--VERBOSE", default=True, type=str2bool)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr_dlpg", default=0.001, type=float)
    parser.add_argument("--eps_dlpg", default=1e-8, type=float)
    parser.add_argument("--n_worker", default=50, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--max_epoch", default=300, type=int)
    parser.add_argument("--n_sim_roll", default=100, type=int)
    parser.add_argument("--sim_update_size", default=64, type=int)
    parser.add_argument("--n_sim_update", default=64, type=int)
    parser.add_argument("--n_sim_prev_consider", default=10, type=int)
    parser.add_argument("--n_sim_prev_best_q", default=50, type=int)
    parser.add_argument("--init_prior_prob", default=0.8, type=float)
    parser.add_argument("--folder", default=0, type=int)
    parser.add_argument("--WANDB", default=False, type=str2bool)
    args = parser.parse_args()
    main(args)
