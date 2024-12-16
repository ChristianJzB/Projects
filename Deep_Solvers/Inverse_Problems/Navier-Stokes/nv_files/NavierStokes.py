from Base.dg import deepGalerkin
import torch

class NavierStokes(deepGalerkin):
    def __init__(self, config,device):
        super().__init__(config,device)

        self.nu = config.nu
        self.chunks = config.chunks
        self.normalization_loss = None
        
    def w_net(self,x_interior):
        x_interior = x_interior.requires_grad_(True)
        pred = self.model(x_interior)

        u = pred[:,0].reshape(-1,1)
        v = pred[:,1].reshape(-1,1)

        u_y = torch.autograd.grad(u, x_interior, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1].reshape(-1,1)
        v_x = torch.autograd.grad(v, x_interior, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 0].reshape(-1,1)

        return v_x - u_y

    def nv_pde(self, x_interior):
        x_interior = x_interior.requires_grad_(True)
        pred = self.model(x_interior)

        u = pred[:,0].reshape(-1,1)
        v = pred[:,1].reshape(-1,1)

        u_x = torch.autograd.grad(u, x_interior, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0].reshape(-1,1)
        v_y = torch.autograd.grad(v, x_interior, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 1].reshape(-1,1)

        w = self.w_net(x_interior)

        # Compute gradients for vorticity (for transport equation)
        w_x = torch.autograd.grad(w, x_interior, grad_outputs=torch.ones_like(w), create_graph=True)[0][:, 0].reshape(-1,1)
        w_y = torch.autograd.grad(w, x_interior, grad_outputs=torch.ones_like(w), create_graph=True)[0][:, 1].reshape(-1,1)
        w_t = torch.autograd.grad(w, x_interior, grad_outputs=torch.ones_like(w), create_graph=True)[0][:, 2].reshape(-1,1)

        w_xx = torch.autograd.grad(w_x, x_interior, grad_outputs=torch.ones_like(w_x), create_graph=True)[0][:, 0].reshape(-1,1)
        w_yy = torch.autograd.grad(w_y, x_interior, grad_outputs=torch.ones_like(w_y), create_graph=True)[0][:, 1].reshape(-1,1)

        # f = 0.1* (torch.sin((x_interior[:,0] + x_interior[:,1])) + 
        #           torch.cos((x_interior[:,0] + x_interior[:,1])))

        # Vorticity transport residual: d_omega/dt + u * d_omega/dx + v * d_omega/dy - nu * laplacian(omega) = f
        transport_residual = w_t + u * w_x + v * w_y - self.nu * (w_xx + w_yy)
        cont = u_x + v_y

        return torch.cat([transport_residual.reshape(-1,1),cont.reshape(-1,1)],axis = 1)
    

    def pde_loss(self,data_interior,tol = 1):

        M = torch.triu(torch.ones((self.chunks, self.chunks)), diagonal=1).T

        nv_pred = self.nv_pde(data_interior)

        nvs_pred, cont = nv_pred[:, 0].view(self.chunks, -1), nv_pred[:, 1].view(self.chunks, -1)

        loss_nvs = torch.mean(nvs_pred**2, dim =1)
        loss_cont = torch.mean(cont**2, dim = 1)

        # Update weights_nvs_cont using exponential decay
        nvs_gamma = torch.exp(-tol * (M @ loss_nvs)).detach()
        cont_gamma = torch.exp(-tol * (M @ loss_cont)).detach()

        gamma = torch.min(nvs_gamma,cont_gamma)

        loss_nvs = torch.mean(loss_nvs * gamma)
        loss_cont = torch.mean(loss_cont * gamma)

        return loss_nvs,loss_cont
    
    def initial_condition(self,output_initial_condition,initial_points,loss_fn):
        w0 = output_initial_condition[:, 0].reshape(-1, 1)
        u0 = output_initial_condition[:, 1].reshape(-1, 1)
        v0 = output_initial_condition[:, 2].reshape(-1, 1)

        u0_v0_pred = self.model(initial_points)
        wo_pred = self.w_net(initial_points)

        loss_u0 = loss_fn(u0_v0_pred[:, 0].view(-1, 1), u0)
        loss_v0 = loss_fn(u0_v0_pred[:, 1].view(-1, 1), v0)
        loss_w0 = loss_fn(wo_pred.view(-1, 1), w0)

        return loss_u0,loss_v0,loss_w0
    
    def losses(self,data_interior,output_initial_condition,initial_points,loss_fn):
        loss_nvs,loss_cont = self.pde_loss(data_interior)
        loss_u0,loss_v0,loss_w0 = self.initial_condition(output_initial_condition,initial_points,loss_fn)

        losses = {"nvs":loss_nvs, "cond":loss_cont, "u0":loss_u0, "v0":loss_v0, "w0":loss_w0}
        return losses