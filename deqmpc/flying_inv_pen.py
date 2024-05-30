import numpy as np
import torch
from rexquad_utils import rk4, deg2rad, Spaces, Spaces_np, w2pdotkinematics_mrp, quat2mrp, euler_to_quaternion, mrp2quat, quatrot, mrp2rot
from torch.func import hessian, vmap, jacrev
import ipdb

class FlyingInvPen_dynamics(torch.nn.Module):
    # think about mrp vs quat for various things - play with Q cost 
    def __init__(self, bsz=1,  mass_q=2.0, mass_p=0.1, J=[[0.01566089, 0.00000318037, 0.0],[0.00000318037, 0.01562078, 0.0], [0.0, 0.0, 0.02226868]], L=0.5, gravity=[0,0,-9.81], motor_dist=0.28, kf=0.0244101, bf=-30.48576, km=0.00029958, bm=-0.367697, quad_min_throttle = 1148.0, quad_max_throttle = 1832.0, ned=False, cross_A_x=0.25, cross_A_y=0.25, cross_A_z=0.5, cd=[0.0, 0.0, 0.0], max_steps=100, dt=0.05, device=torch.device('cpu'), jacobian=False):
        super(FlyingInvPen_dynamics, self).__init__()
        self.m = mass_q + mass_p
        J = np.array(J)
        if len(J.shape) == 1:
            self.J = torch.diag(torch.FloatTensor(J)).unsqueeze(0).to(device)
        else:
            self.J = torch.FloatTensor(J).unsqueeze(0).to(device)
        self.Jinv = torch.linalg.inv(self.J).to(device)
        self.g = torch.FloatTensor(gravity).to(device).unsqueeze(0)
        self.motor_dist = motor_dist
        self.kf = kf
        self.km = km
        self.bf = bf
        self.bm = bm
        self.L = torch.tensor(L)
        self.bsz = bsz
        self.Bf = torch.zeros((1,3)).to(device)
        self.Bf[0,2] = 4*bf
        self.quad_min_throttle = quad_min_throttle
        self.quad_max_throttle = quad_max_throttle
        self.ned = ned
        self.cross_A_x = cross_A_x
        self.cross_A_y = cross_A_y
        self.cross_A_z = cross_A_z
        self.cross_A = torch.FloatTensor(np.array([cross_A_x, cross_A_y, cross_A_y])).to(device).unsqueeze(0)
        self.nx = self.state_dim = 3 + 3*3 + 4
        self.nu = self.control_dim = 4
        self._max_episode_steps = max_steps
        self.bsz = bsz
        self.dt = dt
        self.act_scale = 100.0
        self.u_hover = torch.tensor([(-self.m*gravity[2]-self.bf*4)/self.act_scale/self.kf/4]*4).to(device)
        self.cd = torch.tensor(cd).unsqueeze(0).to(device)
        self.ss = torch.tensor([[1.,1,0], [1.,-1,0], [-1.,-1,0], [-1.,1,0]]).to(device).unsqueeze(0)
        self.ss = self.ss/self.ss.norm(dim=-1).unsqueeze(-1)

        self.device = device
        self.jacobian = jacobian
        self.identity = torch.eye(self.state_dim).to(device)

    def forces(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        m = x[..., 3:6]
        q = mrp2quat(-m)
        kf = 0.0244101
        F = torch.sum(kf*u, dim=-1)
        # g = torch.tensor([0,0,-9.81]).to(x)#self.g
        F = torch.stack([torch.zeros_like(F), torch.zeros_like(F), F], dim=-1)
        if len(m.shape) == 3:
            cd = self.cd.unsqueeze(0)
            cross_A = self.cross_A.unsqueeze(0)
        else:
            cd = self.cd
            cross_A = self.cross_A
        df = -torch.sign(m)*0.5*1.27*(m*m)*cd*cross_A
        # Bf = torch.tensor([0.0, 0.0, -30.48576*4]).to(x.device).unsqueeze(0)
        # df = 0
        f = F + df + quatrot(q, self.m * self.g)  + self.Bf
        # ipdb.set_trace()
        return f

    def moments(self, x, u):
        L = self.motor_dist        
        zeros = torch.zeros_like(u)
        F = self.kf*u#torch.maximum(self.kf*u, zeros)
        M = self.km*u
        tau1 = zeros[...,0]
        tau2 = zeros[...,0]
        tau3 = M[...,0]-M[...,1]+M[...,2]-M[...,3]
        torque = torch.stack([tau1, tau2, tau3], dim=-1)
        ss = self.ss
        if len(x.shape) == 3:
            ss = ss.unsqueeze(0)
        if ss.dtype != x.dtype:
            ss = ss.to(x.dtype)
        torque += torch.cross(self.motor_dist * ss, torch.stack([zeros, zeros, self.kf * u + self.bf], dim=-1), dim=-1).sum(dim=-2)
        return torque

    def wrenches(self, x, u):
        F = self.forces(x, u)
        M = self.moments(x, u)
        return F, M

    def rk4_dynamics(self, x, u):
        # x, u = x.unsqueeze(0), u.unsqueeze(0)
        dt = self.dt
        dt2 = dt / 2.0
        y0 = x
        k1 = self.dynamics_(y0, u)
        k2 = self.dynamics_(y0 + dt2 * k1, u)
        k3 = self.dynamics_(y0 + dt2 * k2, u)
        k4 = self.dynamics_(y0 + dt * k3, u)
        yout = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return yout
    
    def forward(self, x, u):
        return self.rk4_dynamics(x, u)
        
    def get_quad_state(self, x):
        r = x[..., :3]
        m = x[..., 3:6] # mrp2quat?
        v = x[..., 6:9]
        w = x[..., 9:12]
        return r, m, v, w, x[..., 0:12]

    def get_pend_state(self, x):
        a = x[..., 12:14]  # relative position of the pendulum
        a_dot = x[..., 14:16]
        return a, a_dot
        
    def compute_fp_Bp(self, p: torch.Tensor, pdot: torch.Tensor):
        a = p[...,0]
        b = p[...,1]
        psi = torch.sqrt(self.L**2 - a**2 - b**2)
        adot = pdot[...,0]
        bdot = pdot[...,1]
        H1 = 4*bdot**2*(a**2 - self.L**2)
        H2 = -8*adot*bdot*a*b 
        H3 = 4*adot**2*(b**2 - self.L**2)
        H4 = 3*psi**3*self.g[0][2]
        H = H1 + H2 + H3 + H4
        fp0 = a*H/(4*self.L**2*psi**2)
        fp1 = b*H/(4*self.L**2*psi**2)
        fp = torch.stack([fp0, fp1], dim=-1)
        Bp00 = 3*(a**2 - self.L**2)
        Bp01 = 3*a*b
        Bp10 = 3*a*b
        Bp11 = 3*(b**2 - self.L**2)
        Bp02 = 3*a*psi
        Bp12 = 3*b*psi
        Bp0 = torch.stack([Bp00, Bp01, Bp02], dim=-1)
        Bp1 = torch.stack([Bp10, Bp11, Bp12], dim=-1)
        Bp = torch.stack([Bp0, Bp1], dim=-2)/(4*self.L**2)
        return fp, Bp

    def dynamics_(self, x, u):
        u = self.act_scale * u
        p, m, v, w, xq = self.get_quad_state(x) # possibly do mrp2quat
        q = mrp2quat(m)
        F, tau = self.wrenches(xq, u)
        mdot = w2pdotkinematics_mrp(m, w)
        pdot = quatrot(q, v)
        vdot = F/self.m - torch.cross(w, v, dim=-1)
        if len(w.shape) == 3:
            Jinv = self.Jinv.unsqueeze(0)
            J = self.J.unsqueeze(0)
        else:
            Jinv = self.Jinv
            J = self.J
        wdot = (Jinv*(tau - torch.cross(w, (J*(w.unsqueeze(-2))).sum(dim=-1), dim=-1)).unsqueeze(-2)).sum(dim=-1)

        # add the inverted pendulum
        vdot_W = quatrot(q, vdot)
        p2, p2_dot = self.get_pend_state(x)
        fp, Bp = self.compute_fp_Bp(p2, p2_dot)
        p2_ddot = fp + (Bp @ vdot_W.unsqueeze(-1)).squeeze(-1)

        return torch.cat([pdot, mdot, vdot, wdot, p2_dot, p2_ddot], dim=-1)

class FlyingInvPen_dynamics_jac(FlyingInvPen_dynamics):
    def __init__(self, bsz=1,  mass_q=2.0, mass_p=0.1, J=[[0.01566089, 0.00000318037, 0.0],[0.00000318037, 0.01562078, 0.0], [0.0, 0.0, 0.02226868]], L=0.5, gravity=[0,0,-9.81], motor_dist=0.28, kf=0.0244101, bf=-30.48576, km=0.00029958, bm=-0.367697, quad_min_throttle = 1148.0, quad_max_throttle = 1832.0, ned=False, cross_A_x=0.25, cross_A_y=0.25, cross_A_z=0.5, cd=[0.0, 0.0, 0.0], max_steps=100, dt=0.05, device=torch.device('cpu')):
        super(FlyingInvPen_dynamics_jac, self).__init__(bsz, mass_q, mass_p, J, L, gravity, motor_dist, kf, bf, km, bm, quad_min_throttle, quad_max_throttle, ned, cross_A_x, cross_A_y, cross_A_z, cd, max_steps, dt, device, True)
    
    def forward(self, x, u):
        ## use vmap to compute jacobian using autograd.grad
        x = x.unsqueeze(-2).repeat(1, self.state_dim, 1)
        u = u.unsqueeze(-2).repeat(1, self.state_dim, 1)
        out_rk4 = self.rk4_dynamics(x, u)
        out = out_rk4*self.identity[None]
        jac_out = torch.autograd.grad([out.sum()], [x, u])
        
        return out_rk4[:, 0], jac_out

class FlyingInvPen(torch.nn.Module):
    def __init__(self, bsz=1,  mass_q=2.0, mass_p=0.1, J=[[0.01566089, 0.00000318037, 0.0],[0.00000318037, 0.01562078, 0.0], [0.0, 0.0, 0.02226868]], L=0.5, gravity=[0,0,-9.81], motor_dist=0.28, kf=0.0244101, bf=-30.48576, km=0.00029958, bm=-0.367697, quad_min_throttle = 1148.0, quad_max_throttle = 1832.0, ned=False, cross_A_x=0.25, cross_A_y=0.25, cross_A_z=0.5, cd=[0.0, 0.0, 0.0], max_steps=100, dt=0.05, device=torch.device('cpu')):
        super(FlyingInvPen, self).__init__()
        self.dynamics = FlyingInvPen_dynamics(bsz, mass_q, mass_p, J, L, gravity, motor_dist, kf, bf, km, bm, quad_min_throttle, quad_max_throttle, ned, cross_A_x, cross_A_y, cross_A_z, cd, max_steps, dt, device, False)
        # self.dynamics = torch.jit.script(self.dynamics)
        self.dynamics_derivatives = FlyingInvPen_dynamics_jac(bsz, mass_q, mass_p, J, L, gravity, motor_dist, kf, bf, km, bm, quad_min_throttle, quad_max_throttle, ned, cross_A_x, cross_A_y, cross_A_z, cd, max_steps, dt, device)
        # self.dynamics_derivatives = torch.jit.script(self.dynamics_derivatives)
        self.bsz = bsz
        self.nx = self.state_dim = self.dynamics.state_dim
        self.nu = self.control_dim = self.dynamics.control_dim
        self.nq = 9
        self._max_episode_steps = max_steps
        self.num_steps = torch.zeros(bsz).to(device)
        # self.x = self.reset()
        self.device = device
        self.dt = dt
        self.Qlqr = torch.tensor([10.0]*3 + [10.0]*3 + [1.0]*6 + [10.0]*4).to(device)#.unsqueeze(0)
        # self.Qlqr = torch.tensor([10.0]*3 + [0.01]*3 + [1.0]*3 + [0.01]*3).to(device)#.unsqueeze(0)
        self.Rlqr = torch.tensor([1e-8]*self.control_dim).to(device)#.unsqueeze(0)
        self.observation_space = Spaces_np((self.state_dim,))
        # self.max_torque = 18.3
        self.action_space = Spaces_np((self.control_dim,), np.array([18.3]*self.control_dim), np.array([11.5]*self.control_dim)) #12.0
        self.x_window = torch.tensor([5.0,5.0,5.0,deg2rad(70),deg2rad(70),deg2rad(70),0.5,0.5,0.5,0.25,0.25,0.25, L/2, L/2, 0.5, 0.5]).to(device)
        self.targ_pos = torch.zeros(self.state_dim).to(self.device)
        self.spec_id = "FlyingInvPen-v0"
        self.saved_ckpt_name = "cgac_checkpoint_rexquadrotor_eplen100save"

    def forward(self, x, u, jacobian=False):
        if jacobian:
            return self.dynamics_derivatives(x, u)
        else:
            return self.dynamics(x, u)
    
    def step(self, u):
        self.num_steps += 1
        done_inf = torch.zeros(self.bsz).to(self.device, dtype=torch.bool)
        if u.dtype == np.float64 or u.dtype == np.float32:
            u = torch.tensor(u).to(self.x)
            if len(u.shape)==1:
                u = u.unsqueeze(0)
                # dynamics = lambda y: self.dynamics(y, u)
                # self.x = rk4(dynamics, self.x, [0, self.dt])
                self.x = self.dynamics(self.x, u)
                reward = self.reward(self.x, u).cpu().numpy().squeeze()
                if torch.isnan(self.x).sum() or torch.isinf(self.x).sum() or np.isinf(reward) or np.isnan(reward) or reward < -500:
                    x = self.reset()
                    done_inf = True
                    reward = 0
                x_out = self.x.squeeze().detach().cpu().numpy()
            else:
                # dynamics = lambda y: self.dynamics(y, u)
                # self.x = rk4(dynamics, self.x, [0, self.dt])
                self.x = self.dynamics(self.x, u)
                reward = self.reward(self.x, u).cpu().numpy()
                if torch.isnan(self.x).sum() or torch.isinf(self.x).sum() or np.isinf(reward.sum()) or np.isnan(reward.sum()) or reward.sum() < -500:
                    x = self.reset()
                    done_inf = True
                    reward = np.array([0.0])
                x_out = self.x.detach().cpu().numpy()
        elif u.dtype == torch.float32 or u.dtype==torch.float64:
            # dynamics = lambda y: self.dynamics(y, u)
            # self.x = rk4(dynamics, self.x, [0, self.dt])
            self.x = self.dynamics(self.x, u)
            reward = self.reward(self.x, u)
            # if torch.isnan(self.x).sum() or torch.isinf(self.x).sum() or torch.isinf(reward.sum()) or torch.isnan(reward.sum()) or reward.sum() < -500:
            ifcond = torch.logical_or(torch.isnan(self.x).any(dim=-1), torch.logical_or(torch.isinf(self.x).any(dim=-1), torch.logical_or(torch.isinf(reward), torch.logical_or(torch.isnan(reward), (reward < -500)))))
            if ifcond.any():
                self.reset(torch.where(ifcond))
                done_inf[ifcond] = 1
                reward[ifcond] = 0.0
            x_out = self.x
        done = torch.logical_or(self.num_steps >= self._max_episode_steps, done_inf)
        # if np.isinf(reward) or np.isnan(reward):
        #     ipdb.set_trace()
        return x_out, reward, done, {'done_inf':done_inf}

    def stepx(self, x, u):
        done_inf = False
        if u.dtype == np.float64 or u.dtype == np.float32:
            u = torch.tensor(u).to(self.x)
            x = torch.tensor(x).to(self.x)
            if len(u.shape)==1:
                u = u.unsqueeze(0)
                x = x.unsqueeze(0)
                # dynamics = lambda y: self.dynamics(y, u)
                # x = rk4(dynamics, x, [0, self.dt])
                x = self.dynamics(x, u)
                reward = self.reward(x, u).cpu().numpy().squeeze()
                if torch.isnan(x).sum() or torch.isinf(x).sum():
                    self.reset()
                    done_inf = True
                    reward = -1000
                    x = self.x
                x_out = x.squeeze().detach().cpu().numpy()
            else:
                # dynamics = lambda y: self.dynamics(y, u)
                # x = rk4(dynamics, x, [0, self.dt])
                x = self.dynamics(x, u)
                reward = self.reward(x, u).cpu().numpy()
                if torch.isnan(x).sum() or torch.isinf(x).sum():
                    self.reset()
                    done_inf = True
                    reward = np.array([-1000])
                    x = self.x
                x_out = x.detach().cpu().numpy()
        elif u.dtype == torch.float32 or u.dtype==torch.float64:
            # dynamics = lambda y: self.dynamics(y, u)
            x = self.dynamics(x, u)#rk4(dynamics, x, [0, self.dt])
            reward = self.reward(x, u)
            if torch.isnan(x).sum() or torch.isinf(x).sum():
                self.reset()
                done_inf = True
                reward = torch.tensor([-1000]).to(self.x)
                x = self.x
            x_out = x
        done = self.num_steps >= self._max_episode_steps or done_inf
        return x_out, reward, done, {'done_inf':done_inf}

    def reward(self, x, u):
        cost = (((x - self.targ_pos)**2)*self.Qlqr/2).sum(dim=-1)/100 + ((u**2)*self.Rlqr/2).sum(dim=-1)/10
        mask = (cost > 500)
        rew = torch.exp(-cost/2+2)
        rew[mask] = -cost[mask]
        # if torch.isnan(rew).sum() or torch.isinf(rew).sum():
        #     ipdb.set_trace()
        return rew

    def reset_torch(self, reset_ids=None, bsz=None, x_window=None):
        if bsz is None and reset_ids is None:
            bsz = self.bsz
        elif bsz is None:
            bsz = len(reset_ids)
        self.num_steps[reset_ids] = 0
        if x_window is None:
            x_window = self.x_window
        elif len(x_window.shape) == 1:
            x_window = x_window.unsqueeze(0)
        x = (torch.rand((bsz, self.state_dim))*2-1).to(self.x_window)*self.x_window 
        x = torch.cat([x[:,:3], quat2mrp(euler_to_quaternion(x[:, 3:6])), x[:, 6:]], dim=-1) #quat2mrp
        if reset_ids is not None:
            self.x[reset_ids] = x
        else:
            self.x = x
        return x

    def reset(self, reset_ids=None, bsz=None, x_window=None):
        x = self.reset_torch(reset_ids, bsz, x_window)#.detach().cpu().numpy().squeeze()
        return x

if __name__ == "__main__":
    quad = FlyingInvPen(bsz=1, device='cuda').dynamics
    quad_jac = FlyingInvPen(bsz=1, device='cuda').dynamics_derivatives
    scripted_quad = torch.jit.script(quad)
    scripted_quad_jac = torch.jit.script(quad_jac)
    # ipdb.set_trace()
    # x = FlyingInvPen(bsz=1000, device='cuda').reset()
    x = torch.tensor([.0,.0,.0,deg2rad(0.0),deg2rad(0.0),deg2rad(0.0),0.,0.,0.,0.,0.,0., 0.3, 0.0, 0., 0.]).unsqueeze(0).to('cuda')
    x = torch.cat([x[:,:3], quat2mrp(euler_to_quaternion(x[:, 3:6])), x[:, 6:]], dim=-1) #quat2mrp
    # u = torch.tensor([0.0, 0.0, 0.0, 0.0]).unsqueeze(0).repeat(1, 1).to(x).requires_grad_(True)
    u = quad.u_hover.unsqueeze(0).repeat(1, 1).to(x)
    # ipdb.set_trace()
    for i in range(100):
        # x = scripted_quad(x,u)
        x = quad(x,u)
        print(x[:,12:])

    # x.requires_grad = True
    # for i in range(100):
    #     # out = scripted_quad(x,u)
    #     # quad_jac(x,u)
    #     out_jac = scripted_quad_jac(x,u)
    #     # jacobian_computation(x, u, scripted_quad, quad.identity)
    # # ipdb.set_trace()
    # ## time the forward pass
    # # compute jacobian using jacrev
    # # scripted_quad_bsz1 = lambda x, u: scripted_quad(x.unsqueeze(0), u.unsqueeze(0)).squeeze(0)
    # # scripted_quad_jac = torch.jit.script(jacrev(quad.forward))
    # import time
    # start = time.time()
    # # with torch.no_grad():
    # for i in range(1000):
    #     # out = scripted_quad(x,u)
    #     # jacobian_computation(x, u, scripted_quad, quad.identity)
    #     ## compute jacobian
    #     # jacobian = torch.autograd.functional.jacobian(scripted_quad, (x[0],u[0]))#, vectorize=True)
    #     # jacrev = scripted_quad_jac(x[0],u[0])
    #     out_jac = scripted_quad_jac(x,u)
    #     # jacobian = scripted_quad_jac(x,u)


    # end = time.time()
    # print("scripted time taken: ", end-start)

    # ## time the unscripted forward pass
    # for i in range(100):
    #     out = quad_jac(x,u)
    #     # out = quad(x,u)
    # start = time.time()
    # # with torch.no_grad():
    # for i in range(1000):
    #     # out = quad(x,u)
    #     out = quad_jac(x,u)
    # end = time.time()
    # print("unscripted time taken: ", end-start)
    # print("simple test passed")