
import torch
from Base.dg import deepGalerkin
from .FEM_Solver import RootFinder

def k_function(data_domain,w):
    x = data_domain[:,0].reshape(-1,1)
    theta = data_domain[:,1:].reshape(x.shape[0],-1)

    A = torch.sqrt(1 / ( (1/8)*(5 + (w / 2)**2) +  (torch.sin(2*w) / (4*w))*((w / 4)**2 - 1) - (torch.cos(2*w)/8)))
    
    bn =  A*(torch.sin(w*x) + ((w)/(4))*torch.cos(w*x))

    an = torch.sqrt(8 / (w**2 + 16))

    return torch.sum(an*bn*theta,dim=1)


class Elliptic(deepGalerkin):
    def __init__(self, config,device, lam = 1/4):
        super().__init__(config,device)
        self.root_finder = RootFinder(lam, config.KL_expansion)
        self.roots = torch.tensor(self.root_finder.find_roots())
    
    @deepGalerkin.laplace_approx()
    def u(self,x):
        pred = self.model(x)
        return pred.reshape(-1,1)
    
    @deepGalerkin.laplace_approx()
    def elliptic_pde(self, x_interior):
        """ The pytorch autograd version of calculating residual """
        data_domain = x_interior.requires_grad_(True)

        u = self.model(data_domain)

        du = torch.autograd.grad(u, data_domain,grad_outputs=torch.ones_like(u),create_graph=True)[0]

        k = k_function(data_domain,self.roots)
        
        ddu_x = torch.autograd.grad(torch.exp(k).reshape(-1,1)*du[:,0].reshape(-1,1),data_domain, 
            grad_outputs=torch.ones_like(du[:,0].reshape(-1,1)),create_graph=True)[0]
            
        return ddu_x[:,0].reshape(-1,1) + 4*data_domain[:,0].reshape(-1,1)


    def pde_loss(self,data_interior,loss_fn):
        elliptic_pred = self.elliptic_pde(data_interior)
        zeros = torch.zeros_like(elliptic_pred)
        return loss_fn(elliptic_pred,zeros)
    
    def bc_loss(self,bcl_pts,bcr_pts,loss_fn, left_bc=0, right_bc=2):
        u_bcl = self.u(bcl_pts)
        u_bcr = self.u(bcr_pts)

        left_vals = torch.ones_like(u_bcl) * left_bc
        right_vals = torch.ones_like(u_bcr) * right_bc

        loss_ubcl = loss_fn(u_bcl, left_vals)
        loss_ubcr = loss_fn(u_bcr, right_vals)

        return loss_ubcl,loss_ubcr
    
    def losses(self,data_interior,bcl_pts,bcr_pts,loss_fn):
        loss_pde = self.pde_loss(data_interior,loss_fn)
        loss_ubcl,loss_ubcr = self.bc_loss(bcl_pts,bcr_pts,loss_fn)

        losses = {"elliptic":loss_pde, "ubcl":loss_ubcl, "ubcr":loss_ubcr}
        return losses
    