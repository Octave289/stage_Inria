## tentative d'implémentation du modèle stochastique du bassin d'algues
# pas de conditions aux bords ni de condition de non accumulation au niveau 0 (dans la roue à aube)

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

#####################
#constantes :
#####################



def bassin_initial(f, N_lignes, nb_colonnes, l, h):
    delta_x = l/nb_colonnes
    delta_y = h/N_lignes
    X = np.zeros((N_lignes, nb_colonnes))
    for i in range(N_lignes):
        for j in range(nb_colonnes):
            X[i, j] = f(j*delta_x, i*delta_y, l)
    return X

def vecteur_propre(p):
        p_ = np.asarray(p)

        V  = []
        valeurs_propres, vecteurs_propres = np.linalg.eig(p_)

        # Trouver l'indice de la valeur propre la plus proche de 1
        valeurs_propres_translatées = np.real(valeurs_propres - 1)

        vecteur = np.array([0.]*np.shape(p)[0])
        for i in range(len(valeurs_propres_translatées)):
                if np.abs(valeurs_propres_translatées[i]) < 10**(-7):
                        vecteur += vecteurs_propres[:, i] 

        # Normalisation : pour que la somme soit 1 (distribution de probabilité)
        vecteur_normalise = vecteur/sum(v for v in vecteur)

        return vecteur_normalise

def solution_exacte(f, #fonction initiale
                    t, #temps_final
                    N_lignes,
                    nb_colonnes,
                    p,
                    L,
                    H,
                    D,
                    alpha, beta, gamma
                    ):
    if t == 0 :
        return bassin_initial(f, N_lignes, nb_colonnes, L, H)
    delta_x = L/nb_colonnes
    V = vecteur_propre(p)
    N_max = len(alpha)
    X = np.zeros((N_lignes, nb_colonnes))
    for i in range(N_lignes):
        for col in range(nb_colonnes):
            X[i, col] = sum(alpha[k]*V[i]*np.exp(-4*k**2*np.pi**2*D/L**2*t)*np.cos(2*k*np.pi*col*delta_x/L)
                            + beta[i][k]*np.exp(-4*k**2*np.pi**2*D/L**2*t)*np.sin(2*k*np.pi*col*delta_x/L)
                            + gamma[i][k]*np.exp(-(2*k+1)**2*np.pi**2*D/L**2*t)*np.sin((2*k+1)*np.pi*col*delta_x/L)
                            for k in range(N_max))
    
    return X

def verif_CFL(delta_t, delta_x, D, u, N_lignes):
    if delta_x*delta_x <= 2*D*delta_t :
        return False
    else :
        bool = True
        if u != [0]*N_lignes:
            for i in range(N_lignes):
                bool &= delta_x < 2*D/u[i]
    return bool



def mise_a_jour(X1, p, N_lignes, nb_colonnes, L, H, D, u, delta_t):

    delta_x = L / nb_colonnes
    nu = D * delta_t / (delta_x * delta_x)
    K = [u[i] * delta_t / delta_x for i in range(N_lignes)]
    X_tmp = np.copy(X1)

    for i in range(N_lignes):
        for j in range(2, nb_colonnes - 1):
            X_tmp[i, j] = (
                (K[i] + nu) * X1[i, j - 1]
                + (1 - 2 * nu) * X1[i, j]
                + (nu - K[i]) * X1[i, j + 1]
            )

        X_tmp[i, nb_colonnes - 1] = (
            (1 - 2 * nu) * X1[i, nb_colonnes - 1]
            + sum((nu - K[l]) * p[i, l] * X1[l, 0] for l in range(N_lignes))
            + (K[i] + nu) * X1[i, nb_colonnes - 2]
        )

        X_tmp[i, 0] = (
            (K[i] + nu) * X1[i, nb_colonnes - 1]
            + sum((1 - 2 * nu) * p[i, l] * X1[l, 0] for l in range(N_lignes))
            + (nu - K[i]) * X1[i, 1]
        )

        X_tmp[i, 1] = (
            (1 - 2 * nu) * X1[i, 1]
            + sum((K[l] + nu) * p[i, l] * X1[l, 0] for l in range(N_lignes))
            + (nu - K[i]) * X1[i, 2]
        )
    return X_tmp

def mise_a_jour_decentree(X1, p, N_lignes, nb_colonnes, L, H, D, u, delta_t):

    delta_x = L / nb_colonnes
    nu = D * delta_t / (delta_x * delta_x)
    K = [u[i] * delta_t / delta_x for i in range(N_lignes)]
    X_tmp = np.copy(X1)

    for i in range(N_lignes):
        for j in range(2, nb_colonnes - 1):
            X_tmp[i, j] = (
                (K[i] + nu) * X1[i, j - 1]
                + (1 - 2 * nu - K[i]) * X1[i, j]
                + nu * X1[i, j + 1]
            )

        X_tmp[i, nb_colonnes - 1] = (
            (1 - 2 * nu - K[i]) * X1[i, nb_colonnes - 1]
            + sum(nu * p[i, l] * X1[l, 0] for l in range(N_lignes))
            + (K[i] + nu) * X1[i, nb_colonnes - 2]
        )

        X_tmp[i, 0] = (
            (K[i] + nu) * X1[i, nb_colonnes - 1]
            + sum((1 - 2 * nu - K[i]) * p[i, l] * X1[l, 0] for l in range(N_lignes))
            + nu * X1[i, 1]
        )

        X_tmp[i, 1] = (
            (1 - 2 * nu - K[i]) * X1[i, 1]
            + sum((K[l] + nu) * p[i, l] * X1[l, 0] for l in range(N_lignes))
            + nu * X1[i, 2]
        )
    return X_tmp

def mise_a_jour_terme_source(X1, p, N_lignes, nb_colonnes, L, H, D, u, delta_t, 
                             i_s, eps, k_d, k_r, k_h, tau, sigma_H): #terme de transport décentré


    delta_x = L / nb_colonnes
    X_tmp = np.copy(X1)
    nu = D * delta_t / (delta_x * delta_x)
    K = [u[i] * delta_t / delta_x for i in range(N_lignes)]

    for i in range(N_lignes):
        I = i_s*np.exp(-eps*H/N_lignes*i)
        A = 1/(k_d/k_r*tau*(sigma_H*I)**2 + tau*sigma_H*I + 1)
        taux_croiss = k_h*sigma_H*I*A
        for j in range(2, nb_colonnes - 1):
            X_tmp[i, j] = (#on décentre le terme de transport
                (nu  + K[i])*X1[i, j - 1]
                + (1 - 2*nu  - K[i] + taux_croiss * delta_t) * X1[i, j]
                + nu *X1[i, j + 1]
            )

        X_tmp[i, nb_colonnes - 1] = (
            (1 - 2 * nu  - K[i] + taux_croiss *delta_t) * X1[i, nb_colonnes - 1]
            + sum(nu * p[i, l] * X1[l, 0] for l in range(N_lignes))
            + (K[i] + nu) * X1[i, nb_colonnes - 2]
        )

        X_tmp[i, 0] = (
            (K[i] + nu) * X1[i, nb_colonnes - 1]
            + sum((1 - 2 * nu - K[l] + taux_croiss * delta_t) * p[i, l] * X1[l, 0] for l in range(N_lignes))
            + nu * X1[i, 1]
        )

        X_tmp[i, 1] = (
            (1 - 2 * nu - K[i] + taux_croiss *delta_t) * X1[i, 1]
            + sum((K[l] + nu) * p[i, l] * X1[l, 0] for l in range(N_lignes))
            + nu * X1[i, 2]
        )
    return X_tmp

SCHEMAS = {
    "advection_diffusion_centrée": {
        "update": mise_a_jour,
        "dt": lambda dx, D, u, CFL: CFL * dx**2 / D
    },
    "advection_diffusion_décentrée": {
        "update": mise_a_jour_decentree,
        "dt": lambda dx, D, u, CFL: CFL * dx**2 / (2*D + abs(u[0])*dx)
    },
    "advection_diffusion_terme_source": {
        "update": mise_a_jour_terme_source,
        "dt": lambda dx, D, u, CFL: CFL * dx**2 / (2*D + abs(u[0])*dx)
    }
}

def modèle_stochastique(X, t, p, params, schema="advection_diffusion_terme_source",
                        compare=False, f=None, alpha=None, beta=None, gamma=None):
    N_lignes = params["N"]
    nb_colonnes = params["nb_colonnes_default"]
    L = params["L"]
    H = params["H"]
    D = params["D"]          # mu = D
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

    schema_cfg = SCHEMAS[schema]
    Mise_a_jour = schema_cfg["update"]
    delta_t = schema_cfg["dt"](delta_x, D, u, CFL)

    nb_itérations = int(t / delta_t)

    masse_par_couche = [[] for _ in range(N_lignes)]
    if t == 0:
        return np.copy(X)

    nu = D * delta_t / (delta_x * delta_x)
    X1 = np.copy(X)

    erreurs = [] if compare else None

    for step in range(nb_itérations + 1):
        X1 = Mise_a_jour(X1, p, N_lignes, nb_colonnes, L, H, D, u, delta_t, i_s, eps, k_d, k_r, k_h, tau, sigma_H)
        for i in range(N_lignes):
            masse_par_couche[i].append(sum(X1[i, :]))
        if compare:
            X_exa = solution_exacte(f, step * delta_t, N_lignes, nb_colonnes,
                                             p, L, H, D, alpha, beta, gamma)
            erreur = np.max(np.abs(np.array(X_exa).flatten() - X1.flatten()))
            erreurs.append(erreur)

    if t != nb_itérations * delta_t:
        new_delta_t = t - nb_itérations * delta_t
        X1 = Mise_a_jour(X1, p, N_lignes, nb_colonnes, L, H, D, u, new_delta_t, i_s, eps, k_d, k_r, k_h, tau, sigma_H)

    if compare:
        X_exa = solution_exacte(f, t, N_lignes, nb_colonnes,
                                            p, L, H, D, alpha, beta, gamma)
        erreur = np.max(np.abs(np.array(X_exa).flatten() - X1.flatten()))
        erreurs.append(erreur)
        return X1, np.max(erreurs)
    else:
        return X1#, masse_par_couche

# je construit un autre programme similaire mais qui ne fait qu'une itération en temps :

def modèle_stochastique_une_itération(
                        X,  # tableau des valeurs X_i,j  
                        P,
                        params
                        ):
    N_lignes = params["N"]
    nb_colonnes = params["nb_colonnes_default"]
    L = params["L"]
    H = params["H"]
    D = params["D"]          # mu = D
    u = params["u"]
    CFL = params["CFL"]
    return modèle_stochastique(X, CFL*L*L/(D*nb_colonnes*nb_colonnes), P, params)

#il faut à présent appliquer le modèle à quelques exemples que l'on comparera plus tard à l'autre modèle

# On va prendre pour valeurs initiales du tableau X (donc la composition initiale du bassin) celles d'une 
# fonction qu'on aura discrétisée. Par exemple, si l'on prend une fonction affine : f : x, y -> a*x + b*y + c 
# les valeurs des X[i, j] seront celles de la fonction en (i, j)
# prenons pour exemple a = b = c = 1

def modèle_2(X,# tableau des valeurs X_i,j      
            T, #temps_final
            N_lignes,
            nb_colonnes,
            P,
            L,
            H,
            D,
            u,
            CFL):        
    
    delta_x = L/nb_colonnes
    delta_t = CFL*delta_x**2/D
    nb_itérations = int(T/delta_t)
    X0 = X
    if T == 0 :
        return X0
    else:
        nu = D * delta_t / (delta_x * delta_x)
        K = [u[i] * delta_t / (2 * delta_x) for i in range(N_lignes)]

        X1 = np.copy(X0)

        for step in range(nb_itérations + 1):
            X_tmp = np.copy(X1)
            for i in range(N_lignes):
                for j in range(1, nb_colonnes-1):# loin de la roue à aube
                    X_tmp[i, j] = (
                        (K[i] + nu) * X1[i, j - 1]
                        + (1 - 2 * nu) * X1[i, j]
                        + (nu - K[i]) * X1[i, j + 1]
                    )
                X_tmp[i, nb_colonnes - 1] = (
                        (K[i] + nu) * X1[i, nb_colonnes - 2]
                        + (1 - 2 * nu) * X1[i, nb_colonnes - 1]
                        + (nu - K[i]) * X1[i, 0]
                    )
                alpha = sum((nu + K[i])*X1[i, nb_colonnes-1] + (1-2*nu)*X1[i, 0] +(nu - K[i])*X1[i, nb_colonnes-1] for i in range(N_lignes))
                X_tmp[:, 0] = alpha*vecteur_propre(P)


            X1 = X_tmp

        return X1

def modèle_2_une_itération(
                        X,  # tableau des valeurs X_i,j  
                        N_lignes,
                        nb_colonnes,
                        P,
                        L,
                        H,
                        D,
                        u,
                        CFL
                        ):
    return modèle_2(X, CFL*L*L/(D*nb_colonnes*nb_colonnes), N_lignes, nb_colonnes, P, L, H, D, u, CFL)


def X_init(N_lignes, nb_colonnes, L):
    X_ini = np.zeros((N_lignes, nb_colonnes))
    B_tot = 50
    val = B_tot/nb_colonnes
    for j in range(nb_colonnes):
        X_ini[0, j] = val + np.cos(np.pi/nb_colonnes*(j+1/2))
    return X_ini

def X_ini_one_layer(N_lignes, nb_colonnes, k=0, value=1.0):
    X = np.zeros((N_lignes, nb_colonnes))
    X[k, :] = value
    return X

def X_ini_uniform(N_lignes, nb_colonnes,  value=1.0):
    X_ini = np.zeros((N_lignes, nb_colonnes))
    for i in range(N_lignes):
        for j in range(nb_colonnes):
            X_ini[i, j] = value
    return X_ini

def X_ini_exponential(N_lignes, nb_colonnes, decay=0.3):
    X = np.zeros((N_lignes, nb_colonnes))
    for i in range(N_lignes):
        X[i, :] = np.exp(-decay * i)
    return X

def X_ini_horizontal_variation(N_lignes, nb_colonnes):
    X = np.zeros((N_lignes, nb_colonnes))
    for j in range(nb_colonnes):
        X[:, j] = np.sin(2 * np.pi * j / nb_colonnes)
    return X

def X_ini_vertical(N_lignes, nb_colonnes,  value=1.0):
    X = np.zeros((N_lignes, nb_colonnes))
    for i in range(N_lignes):
        X[i, nb_colonnes-1] = value
    return X

def X_ini_random(N_lignes, nb_colonnes, amplitude=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return amplitude * np.random.rand(N_lignes, nb_colonnes)

def X_ini_patch(N_lignes, nb_colonnes, i0=None, j0=None, size=1, value=1.0):
    if i0 is None:
        i0 = N_lignes // 2
    if j0 is None:
        j0 = nb_colonnes // 2

    X = np.zeros((N_lignes, nb_colonnes))
    X[max(0,i0-size):i0+size+1, max(0,j0-size):j0+size+1] = value
    return X


################################################
#optimisation sur les matrices de permutation
################################################


def fonction_objectif(X, t, p, params):
    X = modèle_stochastique(X, t, p, params, "advection_diffusion_terme_source",
                            False, None, None, None, None)
    masse_totale_algues = np.sum(X)
    return masse_totale_algues

def matrix_from_list(liste, N_lignes):
    p = np.zeros((N_lignes, N_lignes))
    for i in range(N_lignes):
        p[i, liste[i]] = 1
    return p


def test_bis(liste, X, t, N_lignes, nb_colonnes, L, H, D, u, CFL,
                     schema): #plus simple, juste pour vérifier le code
    p = matrix_from_list(liste, N_lignes)
    def fonction_bis(X, N):
        masse_totale_algues = 0
        X_ = p@X
        for i in range(len(X)):
            X[i] = (i+1)*X_[i]
            for j in range(len(X[i])):
                masse_totale_algues += X[i][j]
        return masse_totale_algues
    
    masse_totale_algues = fonction_bis(X, N_lignes)
    return(masse_totale_algues, p)

def test(liste, X, t, params):
    N_lignes = params["N"]
    p = matrix_from_list(liste, N_lignes)
    masse_totale_algues = fonction_objectif(X, t, p, params)
    return(masse_totale_algues, p)

def meilleure_matrice_de_permutation(X, t, params,
                     schema="advection_diffusion_terme_source"):
    N_lignes = params["N"]
    meilleure_masse = 0
    meilleure_matrice = np.identity(N_lignes)
    liste_biomasses = []
    liste_matrices = []
    def generer_matrice(n, X, t, params,
                       liste=None):#algorithme de Heap
        nonlocal meilleure_masse, meilleure_matrice
        if liste is None:
            liste = list(range(N_lignes))
        if n==1:
            masse_totale_algues, p = test(liste, X, t, params)
            #liste_matrices.append(p)#
            liste_biomasses.append(masse_totale_algues)
            if masse_totale_algues > meilleure_masse:
                meilleure_masse = masse_totale_algues
                meilleure_matrice = p.copy()
        else:
            generer_matrice(n-1, X, t, params, liste)
        for i in range(n-1):
            if n%2==0:
                liste[i], liste[n-1] = liste[n-1], liste[i]
            else:
                liste[0], liste[n-1] = liste[n-1], liste[0]
            generer_matrice(n-1, X, t, params, liste) 
            
    generer_matrice(N_lignes, X, t, params)
    return (meilleure_masse, meilleure_matrice, liste_biomasses)


def Lagrangien(X, t, N_lignes, nb_colonnes, p, L, H, D, u, CFL, Lambda, schema="advection_diffusion_centrée"):
    constraints = []
    for i in range(N_lignes):
        constraints.append(Lambda[i]*(sum(p[k, i] for k in range(N_lignes)) - 1))
    return (fonction_objectif(X, t, N_lignes, nb_colonnes, p, L, H, D, u, CFL, schema) 
            + sum(constraints))

