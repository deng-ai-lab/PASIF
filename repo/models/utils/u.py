import torch
import numpy as np
from scipy.linalg import expm
from pymanopt.manifolds import SpecialOrthogonalGroup, Euclidean, Product
from pymanopt import Problem
from pymanopt.optimizers import SteepestDescent, ConjugateGradient
from pymanopt.optimizers.line_search import AdaptiveLineSearcher

def kabsch_algorithm_torch(P, Q):
    """
    Kabsch algorithm in PyTorch (supports autograd).
    
    Args:
        P: Source point cloud (N x 3 torch.Tensor).
        Q: Target point cloud (N x 3 torch.Tensor).
    
    Returns:
        R: Optimal rotation matrix (3 x 3 torch.Tensor).
    """
    # Center the point clouds
    P_centered = P - torch.mean(P, dim=0)
    Q_centered = Q - torch.mean(Q, dim=0)
    
    # Compute covariance matrix H = P^T Q
    H = torch.mm(P_centered.T, Q_centered)
    
    # SVD decomposition (U and V are already transposed in PyTorch's SVD)
    U, S, Vh = torch.linalg.svd(H)
    
    # Compute rotation matrix R = V U^T
    R = torch.mm(Vh.T, U.T)
    
    # Handle reflection case (det(R) < 0)
    if torch.det(R) < 0:
        Vh_modified = Vh.clone()
        Vh_modified[:, 2] *= -1  # Flip the last column of V
        R = torch.mm(Vh_modified.T, U.T)
    
    return R

def kabsch_var_torch(x, y, z):
    """
    compute R to minmize ||2(x - yR) - z||_F^2
    
    param:
        x: torch.Tensor, (N, 3) 
        y: torch.Tensor, (N, 3)   
        z: torch.Tensor, (N, 3) 
    
    return:
        R: torch.Tensor, (3, 3)
    """

    A = 2 * x - z  # (N, 3)
    B = 2 * y      # (N, 3)
    
    H = B.T @ A     # (3, 3)
    
    U, S, Vh = torch.linalg.svd(H)
    
    d = torch.det(U @ Vh)
    eye = torch.eye(3, device=x.device)
    eye[2, 2] = torch.sign(d)
    
    R = U @ eye @ Vh
    
    return R

def kabsch_var_mse(x, y, z):

    R = kabsch_var_torch(x, y, z)
    y_aligned = torch.mm(y, R)
    mse = torch.sum((x - y_aligned)**2, dim=-1)
    return mse.mean(), R

def kabsch_var_mse_scale(x, y, z, scale_past=None, R_past=None):

    if scale_past is None:
        scale_past = 1.
    if R_past is None:
        R_past = torch.eye(3, device=x.device).float()
    
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    for i in range(20):
        R = kabsch_var_torch(scale_past*x, scale_past*y, z)
        y_aligned = torch.mm(y, R)
        scale = torch.sum((x-y_aligned)*z) / (2*torch.norm(x-y_aligned)*torch.norm(z))

        current_loss = torch.norm(2*scale*(x - y_aligned) - z, p='fro')**2
        R_diff = torch.norm(R_past - R, p='fro')
        alpha_diff = torch.abs(scale_past - scale)

        R_past = R
        scale_past = scale

    mse = torch.sum((x - y_aligned)**2, dim=-1)
    return mse.mean(), R, scale

def kabsch_mse(P, Q):

    R = kabsch_algorithm_torch(P, Q)
    P_aligned = torch.mm(P - torch.mean(P, dim=0), R) + torch.mean(Q, dim=0)
    mse = torch.sum((P_aligned - Q)**2, dim=-1)

    return mse.mean(), R

class ConstantStepsize:
    """
    Back-tracking line-search based on linesearch.m in the manopt MATLAB
    package.
    """

    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def search(self, objective, manifold, x, d, f0, df0):
        """
        perform a constant step size step in direction d
        Arguments:
            - objective
                objective function to optimise
            - manifold
                manifold to optimise over
            - x
                starting point on the manifold
            - d
                tangent vector at x (descent direction)
            - df0
                directional derivative at x along d
        Returns:
            - stepsize
                norm of the vector retracted to reach newx from x
            - newx
                next iterate suggested by the line-search
        """
        newx = manifold.retraction(x, self.stepsize * d)

        return self.stepsize, newx

from pymanopt.function import autograd
def optimize_R(X, Y, u):
    """
    Optimize the rotation matrix R to minimize the objective function:
    f(R) = tr((X - Y R)(X - Y R)^T) - [tr((X - Y R) u^T)]^2

    Parameters:
        X, Y: m x 3 matrices
        u: m x 3 matrix with ||u||_F = 1 (Frobenius norm = 1)

    Returns:
        Optimal rotation matrix R (3x3)
    """
    
    # Validate input dimensions
    m, n = X.shape
    assert Y.shape == (m, 3) and u.shape == (m, 3), "Input matrix dimensions must match"
    assert np.isclose(np.linalg.norm(u, 'fro'), 1), "u must have Frobenius norm of 1"

    # Define the manifold: SO(3) Lie group (3x3 rotation matrices)
    manifold = SpecialOrthogonalGroup(3)

    @autograd(manifold)
    def cost(R):
        """
        Compute the objective function value at R
        
        Args:
            R: Current rotation matrix (3x3)
            
        Returns:
            Objective function value (scalar)
        """
        residual = X - Y @ R  # Compute residual matrix
        term1 = np.trace(residual @ residual.T)  # First term: squared Frobenius norm
        term2 = np.trace(residual @ u.T) ** 2  # Second term: squared trace
        return term1 - term2

    @autograd(manifold)
    def euclidean_gradient(R):
        """
        Compute the Euclidean gradient of the objective function
        
        Args:
            R: Current rotation matrix (3x3)
            
        Returns:
            Euclidean gradient (3x3 matrix)
        """
        # Gradient of first term: d/dR [tr((X-YR)(X-YR)^T)] = -2 Y^T (X - Y R)
        grad_term1 = -2 * Y.T @ (X - Y @ R)
        
        # Gradient of second term: d/dR [tr((X-YR)u^T)^2] = -2 * tr(...) * (Y^T u)
        grad_term2 = -2 * np.trace((X - Y @ R) @ u.T) * (Y.T @ u)
        
        return grad_term1 + grad_term2

    # Create optimization problem on SO(3) manifold
    problem = Problem(
        manifold=manifold,  # Optimization domain (SO(3))
        cost=cost,  # Objective function
        euclidean_gradient=euclidean_gradient  # Gradient function
    )

    # Initialize solver (steepest descent with default parameters)
    # line_search = ConstantStepsize(1.e-4)
    line_search = AdaptiveLineSearcher()
    solver = SteepestDescent(min_step_size=1.e-4, line_searcher=line_search, verbosity=0)
    # solver = ConjugateGradient(line_searcher=line_search, verbosity=2)
    
    # Solve the optimization problem
    R_opt = solver.run(problem, initial_point=np.eye(3))

    return torch.tensor(R_opt.point).float()


def optimize_SE3(X, Y, u, init_p=None):
    """
    Optimize R and t on SE(3) manifold to minimize:
    f(R,t) = tr((X-YR-1t^T)(X-YR-1t^T)^T) - [tr((X-YR-1t^T)u^T)]^2
    
    Parameters:
        X, Y: m x 3 matrices
        u: m x 3 matrix with ||u||_F = 1
        maxiter: maximum iterations
        
    Returns:
        R_opt: optimal rotation matrix (3x3)
        t_opt: optimal translation vector (3,)
    """
    m = X.shape[0]
    ones = np.ones((m, 1))  # Vector of ones for translation
    
    # Define SE(3) manifold as product of SO(3) and Euclidean(3)
    SO3 = SpecialOrthogonalGroup(3)
    R3 = Euclidean(3)
    manifold = Product([SO3, R3])
    
    @autograd(manifold)
    def cost(R, t):
        # R, t = R_t
        residual = X - Y @ R - ones @ t.reshape(1, 3)
        term1 = np.trace(residual @ residual.T)
        term2 = np.trace(residual @ u.T) ** 2
        return term1 - term2
    
    # Initial guess
    if init_p is None:
        R_init = np.eye(3)
        t_init = np.zeros(3)
        initial_point = (R_init, t_init)
    else:
        initial_point = init_p
    
    # Solve
    problem = Problem(manifold=manifold, cost=cost)
    solver = SteepestDescent(min_step_size=1.e-4, verbosity=2)
    res_opt = solver.run(problem, initial_point=initial_point)
    R_opt, t_opt = res_opt.point
    
    return torch.tensor(R_opt).float(), torch.tensor(t_opt).float()
