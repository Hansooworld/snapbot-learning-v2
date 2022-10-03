import numpy as np
import matplotlib.pyplot as plt
from class_grp import get_anchors_from_traj

class ScalerClass(object):
    """ Generate scale and offset based on running mean and stddev along axis=0
        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
        
        Usage:
            # Scaler
            scaler = Scaler(obs_dim)
            scale, offset = scaler.get()
            obs = (obs - offset) * scale  # center and scale 
            scaler.update(unscaled) # Add to scaler
    """
    def __init__(self, obs_dim):
        self.obs_dim = obs_dim
        self.vars = np.zeros(self.obs_dim)
        self.means = np.zeros(self.obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def reset(self):
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)
        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        x = np.reshape(x, newshape=(-1, self.obs_dim))
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0] # Number of data 
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n # Total number of data
            
    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1 / (np.sqrt(self.vars)+0.1) / 1.0, self.means 

def rollout(env, PID, traj_scale, n_traj_repeat, RENDER=False, PLOT=False):
    env.reset()
    PID.reset()
    L       = traj_scale.shape[0]
    secs    = np.zeros(shape=(L*n_traj_repeat))
    xy_degs = np.zeros(shape=(L*n_traj_repeat, 3))
    x_prev  = env.get_body_com("torso")[0]
    forward_rewards, left_rewards, right_rewards = [], [], []
    cnt     = 0
    for traj_idx in range(n_traj_repeat):
        for tick in range(L):
            PID.update(x_trgt=traj_scale[tick,:], t_curr=env.get_time(), x_curr=env.get_joint_pos_deg())
            _, every_reward, done, rwd_detail = env.step(PID.out())
            if done:
                break
            if env.condition is not None:
                forward_rewards.append(every_reward[0])
                left_rewards.append(every_reward[1])
                right_rewards.append(every_reward[2])
            else:
                forward_rewards.append(every_reward)
            secs[cnt] = env.get_time()
            xy_degs[cnt, :] = np.concatenate((env.get_body_com("torso")[:2],[env.get_heading()]))
            cnt = cnt + 1
            if RENDER:
                env.render()
    traj_secs = secs[:L]
    x_final = env.get_body_com("torso")[0]
    x_diff  = x_final - x_prev

    if PLOT is True:
        t_anchor, x_anchor = get_anchors_from_traj(traj_secs, traj_scale, n_anchor=20)
        figure = plot_snapbot_joint_traj_and_topdown_traj(traj_secs, traj_scale, t_anchor, x_anchor, xy_degs, secs,
                                                figsize=(16,8), title_str='REWARD: [{:.0f}], X_DIFF: [{:.2f}]'.format(sum(forward_rewards), x_diff), tfs=15)
    else:
        figure = None

    return {'secs': secs, 'xy_degs':xy_degs, 'forward_rewards': forward_rewards, 'x_diff': x_diff, 'figure': figure}

def plot_arrow_and_text(
    p             = np.zeros(2),
    deg           = 0.0,
    arrow_len     = 0.01,
    arrow_width   = 0.005,
    head_width    = 0.015,
    head_length   = 0.012,
    arrow_color   = 'r',
    alpha         = 0.5,
    arrow_ec      = 'none',
    arrow_lw      = 1,
    text_str      = 'arrow',
    text_x_offset = 0.0,
    text_y_offset = 0.0,
    tfs           = 10,
    text_color    = 'k',
    bbox_ec       = (0.0,0.0,0.0),
    bbox_fc       = (1.0,0.9,0.8),
    bbox_alpha    = 0.5,
    text_rotation = 0.0
    ):
    """
        Plot arrow with text
    """
    x,y = p[0],p[1]
    u,v = np.cos(deg*np.pi/180.0),np.sin(deg*np.pi/180.0)
    plt.arrow(x=x,y=y,dx=arrow_len*u,dy=arrow_len*v,
              width=arrow_width,head_width=head_width,head_length=head_length,
              color=arrow_color,alpha=alpha,ec=arrow_ec,lw=arrow_lw)
    plt.text(x=x+text_x_offset,y=y+text_y_offset,
             s=text_str,fontsize=tfs,ha='center',va='center',color=text_color,
             bbox=dict(boxstyle="round",ec=bbox_ec,fc=bbox_fc,alpha=bbox_alpha),
             rotation=text_rotation)

def plot_snapbot_joint_traj_and_topdown_traj(
    traj_secs,traj_joints,t_anchor,x_anchor,xy_degs,secs,
    figsize=(16,8),title_str='Snapbot Trajectory (Top View)',tfs=15,SAVE=False,image_name=None
    ):
    """
        Plot Snapbot joint trajectories and topdown-view trajectory
    """
    figure = plt.figure(figsize=figsize)
    plt.subplot(1,2,1); # joint trajectory
    n_joint = traj_joints.shape[1]
    colors = [plt.cm.rainbow(a) for a in np.linspace(0.0,1.0,n_joint)]
    for t_idx in range(n_joint):
        plt.plot(traj_secs,traj_joints[:,t_idx],'-',color=colors[t_idx],label='Joint %d'%(t_idx))
    if t_anchor is not None and x_anchor is not None:
        for t_idx in range(n_joint):
            plt.plot(t_anchor,x_anchor[:,t_idx],'o',color=colors[t_idx])
    plt.xlabel('Time (sec)',fontsize=13)
    plt.ylabel('Position (deg)',fontsize=13)
    plt.legend(fontsize=10,loc='upper left')
    plt.title('Joint Trajectories',fontsize=tfs)

    plt.subplot(1,2,2); # top-down view
    p_torsos = xy_degs[:,:2]
    z_degs   = xy_degs[:,2]
    plt.plot(p_torsos[:,0],p_torsos[:,1],'-',lw=2,color='k')
    n_arrow = 10
    colors = [plt.cm.Spectral(i) for i in np.linspace(1,0,n_arrow)]
    max_tick = p_torsos.shape[0]
    scale = 10.0*max(p_torsos.max(axis=0)-p_torsos.min(axis=0))
    for t_idx,tick in enumerate(
        np.linspace(start=0,stop=max_tick-1,num=n_arrow).astype(np.int32)):
        p,deg    = p_torsos[tick,0:2],z_degs[tick]
        text_str = '[%d] %.1fs'%(tick,secs[tick])
        plot_arrow_and_text(p=p,deg=deg,text_str=text_str,tfs=8,
                            arrow_color=colors[t_idx],arrow_ec='none',
                            alpha=0.2,text_x_offset=0.0,text_y_offset=0.0,
                            arrow_len=0.01*scale,arrow_width=0.005*scale,
                            head_width=0.015*scale,head_length=0.012*scale,
                            text_rotation=deg,bbox_fc='w',bbox_ec='k')
    # Initial and final pose
    plot_arrow_and_text(p=p_torsos[0,0:2],deg=z_degs[0],text_str='Start',tfs=15,
                        arrow_color=colors[0],arrow_ec='k',arrow_lw=2,
                        alpha=0.9,text_x_offset=0.0,text_y_offset=0.0075*scale,
                        arrow_len=0.01*scale,arrow_width=0.005*scale,
                        head_width=0.015*scale,head_length=0.012*scale,
                        text_rotation=z_degs[0],bbox_fc='w',bbox_ec='k',text_color='b')
    plot_arrow_and_text(p=p_torsos[-1,0:2],deg=z_degs[-1],text_str='Final',tfs=15,
                        arrow_color=colors[-1],arrow_ec='k',arrow_lw=2,
                        alpha=0.9,text_x_offset=0.0,text_y_offset=0.0075*scale,
                        arrow_len=0.01*scale,arrow_width=0.005*scale,
                        head_width=0.015*scale,head_length=0.012*scale,
                        text_rotation=z_degs[-1],bbox_fc='w',bbox_ec='k',text_color='b')
    plt.axis('equal'); plt.grid('on')
    plt.title('Top-down Torso Trajectory',fontsize=tfs)
    plt.suptitle(title_str,fontsize=tfs)
    if SAVE:
        plt.savefig(image_name)
    return figure