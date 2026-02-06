import numpy as np
import torch

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

def modele_stochastique_torch(X, t, P, params):
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
        X1 = mise_a_jour_terme_source_torch(X1, P, N_lignes, nb_colonnes, L, H, D, u, delta_t, i_s, eps, k_d, k_r, k_h, tau, sigma_H)

    # --- Dernier pas fractionnaire ---
    if t != nb_iterations * delta_t:
        new_delta_t = t - nb_iterations * delta_t
        X1 = mise_a_jour_terme_source_torch(X1, P, N_lignes, nb_colonnes, L, H, D, u, new_delta_t, i_s, eps, k_d, k_r, k_h, tau, sigma_H)

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
    X1 = modele_stochastique_torch(X0, t, p, params)
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

def solve(X_init, t, params, start_with_matrix=False, M=[[]]):
    N_lignes = params["N"]
    if start_with_matrix:
        Q = torch.nn.Parameter(torch.tensor(M))
    else:
        Q = torch.nn.Parameter(torch.randn(N_lignes, N_lignes))
    optimizer = torch.optim.Adam([Q], lr=1e-1)
    biomass_list = []
    for it in range(150):
        optimizer.zero_grad()
        P = torch.softmax(Q, dim=0)
        loss = -fonction_objectif_torch(X_init, t, P, params)
        biomass_list.append(-loss.item())
        loss.backward()
        optimizer.step()

        if it % 50 == 0:
            print(it, -loss.item())
    return P, biomass_list

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
            X_init, t, N_lignes, nb_colonnes,
            P, L, H, mu, u, CFL)

        biomass_list.append(-loss.item())

        loss.backward()
        optimizer.step()

        if it % 25 == 0:
            print(it, -loss.item())

    return torch.softmax(Q, dim=0), biomass_list

def lr_finder(
    X_init, t, N_lignes, nb_colonnes, L, H, mu, u, CFL,
    lr_start=1e-6,
    lr_end=1e-1,
    n_iter=250
):
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
            X_init, t, N_lignes, nb_colonnes, P, L, H, mu, u, CFL
        )

        loss.backward()
        optimizer.step()

        lrs.append(lr)
        losses.append(-loss.item())

    return np.array(lrs), np.array(losses)


#####autre : méthode de Newton à la main, avec GMRES

def matrix_to_vector(p):
    return p.reshape(-1)

def vector_to_matrix(v, n):
    return v.reshape((n, n))

def J(p):
    #return fonction_objectif_torch(X, t, p, params)
    return(sum(sum(p@p)))

def hessian(p, h):
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

def test():
    return 5



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




