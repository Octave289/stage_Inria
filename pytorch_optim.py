# -*- coding: utf-8 -*-

import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import modele 
from scipy.optimize import minimize

#Q = torch.nn.Parameter(torch.randn(N_lignes, N_lignes))
#P = torch.softmax(Q, dim=0)

def mise_a_jour_terme_source_torch(X, P, N_lignes, nb_colonnes, L, H, D, u, delta_t, i_s, eps, k_d, k_r, k_h, tau, sigma_H):
    delta_x = L / nb_colonnes
    nu = D * delta_t / (delta_x * delta_x)

    K = torch.tensor(u, dtype=X.dtype, device=X.device) * delta_t / delta_x 
    X_new = X.clone()

    for i in range(N_lignes):
        I = i_s * np.exp(-eps * H / N_lignes * i)
        A = 1.0 / (k_d/k_r * tau * (sigma_H*I)**2 + tau*sigma_H*I + 1)
        taux_croiss = k_h * sigma_H * I * A

        for j in range(2, nb_colonnes - 1):
            X_new[i, j] = (
                (nu + K[i]) * X[i, j-1]
                + (1 - 2*nu - K[i] + taux_croiss*delta_t) * X[i, j]
                + nu * X[i, j+1]
            )

        X_new[i, -1] = (
            (1 - 2*nu - K[i] + taux_croiss*delta_t) * X[i, -1]
            + nu * torch.sum(P[i, :] * X[:, 0])
            + (K[i] + nu) * X[i, -2]
        )

        X_new[i, 0] = (
            (K[i] + nu) * X[i, -1]
            + torch.sum((1 - 2*nu - K + taux_croiss*delta_t) * P[i, :] * X[:, 0])
            + nu * X[i, 1]
        )

        X_new[i, 1] = (
            (1 - 2*nu - K[i] + taux_croiss*delta_t) * X[i, 1]
            + torch.sum((K + nu) * P[i, :] * X[:, 0])
            + nu * X[i, 2]
        )

    return X_new
def mise_a_jour_terme_source_vectorial(X, P, N_lignes, nb_colonnes, L, H, D, u, delta_t, i_s, eps, k_d, k_r, k_h, tau, sigma_H):
    
    M = torch.zeros((nb_colonnes, nb_colonnes))
    delta_x = L/nb_colonnes
    nu = D * delta_t / (delta_x * delta_x)
    K = u[0] * delta_t / delta_x

    X_new = X.clone()
    X_temp = X.clone()

    for i in range(nb_colonnes):
        M[i, i] = 1 - 2*nu - K
        if i<nb_colonnes-1:
            M[i, i+1] = nu 
        if i>0:
            M[i, i-1] = nu + K
    M[nb_colonnes-1, 0] = nu 
    M[0, nb_colonnes-1] = nu + K
    X_temp_col0 = P @ X_temp[:, 0]   # nouveau tensor
    X_temp = X_temp.clone()           # clone pour casser la vue
    X_temp[:, 0] = X_temp_col0

    X_new = X_temp@M.T

    for i in range(N_lignes):
        I = i_s * np.exp(-eps * H / N_lignes * i)
        A = 1.0 / (k_d/k_r * tau * (sigma_H*I)**2 + tau*sigma_H*I + 1)
        taux_croiss = k_h * sigma_H * I * A
        X_new[i, :] = X_new[i, :] + delta_t*taux_croiss*X_temp[i, :]
    
    return X_new



def modele_stochastique_torch(X, t, P, params, mise_a_jour=mise_a_jour_terme_source_torch):
    # --- Paramètres ---
    N_lignes = params["N"]
    nb_colonnes = params["nb_colonnes_default"]
    L = params["L"]
    H = params["H"]
    D = params["D"]          
    u = params["u"]
    CFL = params["CFL"]
    i_s = params["I_s"]
    eps = params["epsilon"]
    k_d = params["k_d"]
    k_r = params["k_r"]
    k_h = params["k_h"]
    tau = params["tau"]
    sigma_H = params["sigma_H"]
    # --- Discrétisation ---
    delta_x = L / nb_colonnes
    delta_t = CFL * delta_x**2 / (2 * D + abs(u[0]) * delta_x)

    nb_iterations = int(t / delta_t)

    if t == 0:
        return X

    X1 = X.clone()

    # --- Boucle temporelle ---
    for _ in range(nb_iterations + 1):
        X1 = mise_a_jour(X1, P, N_lignes, nb_colonnes, L, H, D, u, delta_t, i_s, eps, k_d, k_r, k_h, tau, sigma_H)

    # --- Dernier pas fractionnaire ---
    if t != nb_iterations * delta_t:
        new_delta_t = t - nb_iterations * delta_t
        X1 = mise_a_jour(X1, P, N_lignes, nb_colonnes, L, H, D, u, new_delta_t, i_s, eps, k_d, k_r, k_h, tau, sigma_H)

    return X1

def X_ini_torch(N_lignes, nb_colonnes):
    X_ini = torch.zeros((N_lignes, nb_colonnes))
    for i in range(N_lignes):
        for j in range(nb_colonnes):
            X_ini[i, j] = torch.cos(torch.tensor((i + j) / nb_colonnes))
    return X_ini

def X_ini_one_layer_torch(N_lignes, nb_colonnes, k=0, value=1.0):
    X = torch.zeros((N_lignes, nb_colonnes))
    X[k, :] = torch.tensor(value)
    return X

def X_ini_uniform_torch(N_lignes, nb_colonnes, value=1.0):
    X = torch.zeros((N_lignes, nb_colonnes))
    for i in range(N_lignes):
        X[i, :] = torch.tensor(value)
    return X

def fonction_objectif_torch(X0, t, p, params):
    X1 = modele_stochastique_torch(X0, t, p, params, mise_a_jour_terme_source_vectorial)
    return torch.sum(X1)

def fonction_objectif_torch_fast(
    X0, t, p, params,
    detach_every=10
):
    N_lignes = params["N"]
    nb_colonnes = params["nb_colonnes_default"]
    L = params["L"]
    H = params["H"]
    mu = params["D"]          # mu = D
    u = params["u"]
    CFL = params["CFL"]
    i_s = params["I_s"]
    eps = params["epsilon"]
    k_d = params["k_d"]
    k_r = params["k_r"]
    k_h = params["k_h"]
    tau = params["tau"]
    sigma_H = params["sigma_H"]
    delta_x = L / nb_colonnes
    delta_t = CFL * delta_x**2 / (2*mu + abs(u[0])*delta_x)

    if t == 0:
        return torch.sum(X0)

    nb_iterations = int(t / delta_t)

    X = X0.clone()

    for step in range(nb_iterations):
        X = mise_a_jour_terme_source_torch(
            X, p, N_lignes, nb_colonnes,
            L, H, mu, u, delta_t, i_s, eps, k_d, k_r, k_h, tau, sigma_H
        )

        # TRONCATURE DU GRADIENT TEMPOREL
        if step % detach_every != 0:
            X = X.detach()

    # dernier pas de temps (si t n'est pas multiple de delta_t)
    reste = t - nb_iterations * delta_t
    if reste > 0:
        X = mise_a_jour_terme_source_torch(
            X, p, N_lignes, nb_colonnes,
            L, H, mu, u, reste, i_s, eps, k_d, k_r, k_h, tau, sigma_H
        )

    return torch.sum(X)

def fonction_test(X0, p):
    X = torch.matmul(p, X0)
    X[1, :] *= torch.tensor(2.)
    X[2, :] *= torch.tensor(3.)
    return torch.sum(X)

def solve_test(X_init):
    Q = torch.nn.Parameter(torch.randn(3, 3))
    optimizer = torch.optim.Adam([Q], lr=1e-2)

    for it in range(1000):
        optimizer.zero_grad()

        P = torch.softmax(Q, dim=0)
        loss = -fonction_test(X_init, P)

        loss.backward()
        optimizer.step()

        if it % 50 == 0:
            print(it, -loss.item())
    return P


##### test gradient

def test_gradient(X_init, t, params):
    N_lignes = params["N"]
    errors = []
    for k in range(1, 5):
        epsi = 10**(-k)
        Q = torch.nn.Parameter(torch.randn(N_lignes, N_lignes))
        H = torch.softmax(torch.randn(N_lignes, N_lignes), dim=0)
        P = torch.softmax(Q, dim=0)
        loss = -fonction_objectif_torch(X_init, t, P, params)
        loss.backward()
        grad = torch.softmax(Q.grad, dim=0)
        error = torch.sum(grad*(H - P)
                    - (fonction_objectif_torch(X_init, t, (1-epsi)*P + epsi*H, params) 
                    - fonction_objectif_torch(X_init , t, (1+epsi)*P - epsi*H, params))/(2*epsi))
        errors.append(error.detach())
    return(errors)



##### ADAM method

def solve(X_init, t, params, l_r=0.9, start_with_matrix=False, M=[[]]):
    N_lignes = params["N"]
    if start_with_matrix:
        Q = torch.nn.Parameter(torch.tensor(M))
    else:
        Q = torch.nn.Parameter(torch.randn(N_lignes, N_lignes))
    optimizer = torch.optim.Adam([Q], lr=l_r)
    biomass_list = []
    for it in range(15):
        optimizer.zero_grad()
        P = torch.softmax(Q, dim=0)
        loss = -fonction_objectif_torch(X_init, t, P, params)
        biomass_list.append(-loss.item())
        loss.backward()
        optimizer.step()
        print(it, -loss.item())
    return P, biomass_list

def solve_bistochastique(X_init, t, params, max_iter=30, penalty_factor=1000, lr=0.2, start_with_matrix=False, M=torch.eye(10)):
    N_lignes = params["N"]
    if start_with_matrix:
        Q = torch.nn.Parameter(M)
    else:
        Q = torch.nn.Parameter(torch.randn(N_lignes, N_lignes))
    optimizer = torch.optim.Adam([Q], lr=lr, maximize=True)
    biomass_list = []
    for it in range(max_iter):
        optimizer.zero_grad()
        P = torch.softmax(Q, dim=0)
        penalties = 0
        row_sums = P.sum(dim=1)
        penalties = ((row_sums - 1)**2).sum()
        loss = fonction_objectif_torch(X_init, t, P, params) - penalty_factor*penalties
        print(P)
        print("obj:", loss + penalty_factor*penalties,
      "pen:", penalties)
        biomass_list.append(loss + penalty_factor*penalties)
        loss.backward()
        optimizer.step()
    return P, biomass_list

def inverse_softmax(M): #M de permutation
    M_ = M
    for i in range(len(M)):
        for j in range(len(M)):
            if M[i][j] == 0. :
                M_[i][j] = -10.
            if M[i][j] == 1. :
                M[i][j] = 0.
    return M_


def fonction_objectif_optimize_test(X_init, t, p_list, params):
        N = params["N"]
        p = vector_to_matrix(p_list, N)

        X = modele.modèle_stochastique(X_init, t, p, params, "advection_diffusion_terme_source",
                                False, None, None, None, None)
        masse_totale_algues = np.sum(X)
        print(masse_totale_algues)
        return -masse_totale_algues

def solve_optimize(X_init, t, params, p0): 

    def fonction_objectif_optimize(p_list):
        N = params["N"]
        p = vector_to_matrix(p_list, N)

        X = modele.modèle_stochastique(X_init, t, p, params, "advection_diffusion_terme_source",
                                False, None, None, None, None)
        masse_totale_algues = np.sum(X)
        print(masse_totale_algues)
        return -masse_totale_algues
    
    constraints = []
    N = params["N"]
    for i in range(N):
        constraints.append({'type': 'eq', 'fun' : lambda x, i=i: np.sum(x[N*i:N*(i+1)]) - 1})
        for j in range(N):
            constraints.append({'type': 'ineq', 'fun' : lambda x, i=i, j=j: x[N*i + j]})
    for j in range(N-1):
        constraints.append({'type': 'eq', 'fun' : lambda x, j=j: np.sum(x[N*i + j] for i in range(N)) - 1})
    

    return minimize(fonction_objectif_optimize, p0, constraints=constraints)
    
def solve_fast(
    X_init, t, N_lignes, nb_colonnes, L, H, mu, u, CFL,
    n_iter=150,
    detach_every=10
):
    Q = torch.nn.Parameter(torch.randn(N_lignes, N_lignes))
    optimizer = torch.optim.Adam([Q], lr=1e-2)

    biomass_list = []

    for it in range(n_iter):
        optimizer.zero_grad()

        # 1️⃣ softmax calculé UNE SEULE FOIS
        P = torch.softmax(Q, dim=0)

        # 2️⃣ appel à une version modifiée de l’objectif
        loss = -fonction_objectif_torch_fast(
            X_init, t, params)

        biomass_list.append(-loss.item())

        loss.backward()
        optimizer.step()

        if it % 25 == 0:
            print(it, -loss.item())

    return torch.softmax(Q, dim=0), biomass_list

def lr_finder(
    X_init, t, params,
    lr_start=1e-6,
    lr_end=1e-1,
    n_iter=250
):
    N_lignes = params["N"]
    Q = torch.nn.Parameter(torch.randn(N_lignes, N_lignes))

    optimizer = torch.optim.Adam([Q], lr=lr_start)

    # croissance exponentielle du LR
    lr_mult = (lr_end / lr_start) ** (1 / (n_iter - 1))

    lrs = []
    losses = []

    for it in range(n_iter):
        optimizer.zero_grad()

        # mettre à jour le LR
        lr = lr_start * (lr_mult ** it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        P = torch.softmax(Q, dim=0)

        loss = -fonction_objectif_torch(
            X_init, t, P, params
        )

        loss.backward()
        optimizer.step()

        lrs.append(lr)
        losses.append(-loss.item())

    return np.array(lrs), np.array(losses)


#####autre : méthode de Newton avec gradient conjugué

def matrix_to_vector(p):
    return p.reshape(-1)

def vector_to_matrix(v, n):
    return v.reshape((n, n))

def conjugate_gradient(Av, b, x0, tol, max_iter):
    x = x0
    r = b - Av(x0)
    p = r.clone()
    step = 0
    while torch.norm(r) > tol and step < max_iter:
        alpha = r@r/(p@Av(p))
        x = x + alpha*p
        r1 = r - alpha*Av(p)
        beta = r1@r1/(r@r)
        p = r1 + beta*p
        r = r1
        step += 1
    return x


def hvp(loss, x, v):
    grad = torch.autograd.grad(loss, x, create_graph=True)[0]
    hv = torch.autograd.grad(
        grad, x, grad_outputs=v, retain_graph=True
    )[0]
    return hv

def newton_cg_step(loss, x, tol=1e-5, max_iter=20):
    grad = torch.autograd.grad(loss, x, create_graph=True)[0]

    def Av(v):
        return hvp(loss, x, v)

    # CG pour résoudre H p = -grad
    p0 = torch.zeros_like(x)
    p = conjugate_gradient(Av, -grad, p0, tol, max_iter)
    return p


def Newton(f, x0, max_iter=20, tol=10e-5):
    x = x0
    for k in range(max_iter):
        loss_val = f(x)
        grad = torch.autograd.grad(loss_val, x, create_graph=True)[0]
        if torch.norm(grad) < tol:
            break

        p = newton_cg_step(loss_val, x)
        x = (x + p).detach().requires_grad_(True)  
    return x, loss_val

def softmax(x, axis=0):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def J(p):
    #return fonction_objectif_torch(X, t, p, params)
    return(sum(sum(p@p)))



def hessian_approx(p, h):
    n = p.shape[0]
    v = matrix_to_vector(p)
    N = n * n
    H = np.zeros((N, N))

    def J_vec(v):
        return J(vector_to_matrix(v, n))

    for i in range(N):
        for j in range(N):
            ei = np.zeros(N); ei[i] = 1
            ej = np.zeros(N); ej[j] = 1

            H[i, j] = (
                J_vec(v + h*ei + h*ej)
                - J_vec(v + h*ei)
                - J_vec(v + h*ej)
                + J_vec(v)
            ) / h**2
    return H





#####autre méthode (Newton + descente gradient)

def LBFGS(X_init, t, N_lignes, nb_colonnes, L, H, mu, u, CFL):
    Q = torch.nn.Parameter(torch.randn(N_lignes, N_lignes))
    optimizer = torch.optim.LBFGS([Q], lr=1e-2)

    for it in range(250):
        optimizer.zero_grad()

        P = torch.softmax(Q, dim=0)
        loss = -fonction_objectif_torch(X_init, t, N_lignes, nb_colonnes, P, L, H, mu, u, CFL)

        loss.backward()
        optimizer.step()

        if it % 50 == 0:
            print(it, -loss.item())
    return P




