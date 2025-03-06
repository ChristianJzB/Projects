import torch
from Base.dg import deepGalerkin


class Heat(deepGalerkin):
    def __init__(self, config,device):
        super().__init__(config,device)

    @deepGalerkin.laplace_approx()
    def u(self,x):
        pred = self.model(x)
        return pred.reshape(-1,1)
    
    @deepGalerkin.laplace_approx()
    def heat_pde(self, x_interior):
        """ The pytorch autograd version of calculating residual """
        data_domain = x_interior.requires_grad_(True)

        u = self.u(data_domain)
    
        du = torch.autograd.grad(u, data_domain,grad_outputs=torch.ones_like(u),create_graph=True)[0]

        ddu_x = torch.autograd.grad(du[:,0],data_domain,
                                    grad_outputs=torch.ones_like(du[:,0]),create_graph=True)[0]
        
        f = du[:,1].reshape(-1,1) - data_domain[:,2].reshape(-1,1)*ddu_x[:,0].reshape(-1,1)- torch.sin(5*torch.pi*data_domain[:,0].reshape(-1,1))
        return f

    def pde_loss(self,data_interior,loss_fn):
        elliptic_pred = self.heat_pde(data_interior)
        zeros = torch.zeros_like(elliptic_pred)
        return loss_fn(elliptic_pred,zeros)
    
    def bc_loss(self,bcl_pts,bcr_pts,loss_fn, left_bc=0, right_bc=0):
        u_bcl = self.u(bcl_pts)
        u_bcr = self.u(bcr_pts)

        left_vals = torch.ones_like(u_bcl) * left_bc
        right_vals = torch.ones_like(u_bcr) * right_bc

        loss_ubcl = loss_fn(u_bcl, left_vals)
        loss_ubcr = loss_fn(u_bcr, right_vals)

        return loss_ubcl,loss_ubcr
    
    def ic_loss(self,ic_pts,loss_fn):
        u_ic = self.u(ic_pts)
        left_vals = 4*torch.sin(3*torch.pi*ic_pts[:,0].reshape(-1,1)) + 9*torch.sin(7*torch.pi*ic_pts[:,0].reshape(-1,1))
        loss_uic = loss_fn(u_ic, left_vals)
        return loss_uic
    
    def losses(self,data_interior,ic_pts,bcl_pts,bcr_pts,loss_fn):
        loss_pde = self.pde_loss(data_interior,loss_fn)
        loss_uic = self.ic_loss(ic_pts,loss_fn)
        loss_ubcl,loss_ubcr = self.bc_loss(bcl_pts,bcr_pts,loss_fn)

        losses = {"heat":loss_pde,"uic":loss_uic ,"ubcl":loss_ubcl, "ubcr":loss_ubcr}
        return losses