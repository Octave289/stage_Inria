import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from modele import *

def dessin_bassin(X):
    fig, ax = plt.subplots()
    X_show = np.copy(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            j0 = j + X.shape[1] // 2
            if j0 > X.shape[1] - 1:
                j0 %= X.shape[1]
            X_show[i, j] = X[i, j0]
    im = ax.imshow(X_show, cmap="Greens", interpolation=None)
    ax.set_xticks(np.arange(-.5, X.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, X.shape[0], 1), minor=True)
    ax.grid(which='minor', linewidth=1)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    for i in range(X.shape[0]):
        ax.text(X.shape[1] // 2, i, "R", ha="center", va="center", color="red", fontsize=6)
    plt.show()


def anim(
        modèle, 
        N_lignes,
        nb_colonnes,
        P,
        L,
        H,
        mu,
        u,
        CFL):
    
    X_anim = np.copy(X_ini(N_lignes, nb_colonnes))

    fig, ax = plt.subplots()
    im = ax.imshow(X_anim, cmap="Greens", interpolation=None, aspect="equal")

    ax.set_xticks(np.arange(-.5, nb_colonnes, 1), minor=True)
    ax.set_yticks(np.arange(-.5, N_lignes, 1), minor=True)
    ax.grid(which='minor', linewidth=1)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    for i in range(N_lignes):
        ax.text(nb_colonnes // 2, i, "R", ha="center", va="center", color="red", fontsize=6)

    def update(frame):
        nonlocal X_anim
        X_anim = modèle(X_anim,
                        N_lignes,
                        nb_colonnes,
                        P,
                        L,
                        H,
                        mu,
                        u,
                        CFL)
        X_show = np.copy(X_anim)
        for i in range(N_lignes):
            for j in range(nb_colonnes):
                j0 = j + nb_colonnes // 2
                if j0 > nb_colonnes - 1:
                    j0 %= nb_colonnes
                X_show[i, j] = X_anim[i, j0]
        im.set_array(X_show)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=3000, interval=100, blit=False)
    return ani


def show_ani(ani):
    plt.show()

# On peut bien évidemment tester le modèle avec différentes matrices de mélange, différentes vitesse en fonction des couches 
#l'idée est d'étudier la convergence du modèle quand les delta_x et delta_t tendent simultanément vers 0 ...

##faire un graphe de la fonction au temps T pour voir son évolution en fonction du pas de temps et d'espace

def vecteur_propre(P):
        P_ = np.asarray(P)

        V  = []
        valeurs_propres, vecteurs_propres = np.linalg.eig(P_)

        # Trouver l'indice de la valeur propre la plus proche de 1
        valeurs_propres_translatées = np.real(valeurs_propres - 1)

        vecteur = np.array([0., 0., 0.])
        for i in range(len(valeurs_propres_translatées)):
                if np.abs(valeurs_propres_translatées[i]) < 10**(-7):

                        vecteur += vecteurs_propres[:, i] 

        # Normalisation : pour que la somme soit 1 (distribution de probabilité)
        vecteur_normalise = vecteur/sum(v for v in vecteur)

        return vecteur_normalise

def graphe(k, #numéro de la ligne à considérer
            X,
            l
           ):
    nb_colonnes = X.shape[1]
    delta_x = l/nb_colonnes
    if nb_colonnes%2 == 0:
        absc = [delta_x*(j - nb_colonnes//2) for j in range(1, nb_colonnes + 1)]
    else :
        absc = [delta_x*(j - nb_colonnes//2) for j in range(nb_colonnes)]
    ord = [X[k + 1, j] for j in range(nb_colonnes//2 + 1, nb_colonnes)] + [X[k + 1, j] for j in range(0, nb_colonnes//2 + 1)]
    plt.plot(absc, ord, label=f"nb de colonnes : {nb_colonnes}") 
    plt.title(f"Ligne {k}")
    plt.legend()
    plt.show()

#graphe(X_ini, 2)

def erreur(k,
           temps_final,
           J,
           N_lignes,
           P,
           L, 
           H, 
           mu, 
           u, 
           CFL
             # = nb de colonnes, qui va donner la valeur de delta_x
            ):#on prend par défaut la ligne 0 du bassin et l'instant 30 (arbitraire)
    # mais delta_t dépend de nb_colonnes donc il faut que n soit proportionnel à nb_colonnes*nb_colonnes pour rester 
    # au même instant (car delta_t = L*L/(4*mu*nb_colonnes*nb_colonnes))
    X1 = modèle_stochastique(X_ini(J), temps_final, N_lignes, J, 4*temps_final*mu*J*J/(L*L))
    X2 = modèle_stochastique(X_ini(2*J), temps_final, N_lignes, 2*J)
    erreur = np.abs(X1[k, 0] - X2[k, 0])
    for j in range(1, J):
        if np.abs(X1[k, j] - X2[k, 2*j]) > erreur:
            erreur = np.abs(X1[k, j] - X2[k, 2*j])
    
    return(erreur)

def graphe_log_erreur(num_ligne, 
                    temps_final,
                    N_lignes,
                    P,
                    L, 
                    H, 
                    mu, 
                    u, 
                    CFL): 
    absc= []
    ords = []
    X1 = modèle_stochastique(X_ini, temps_final, N_lignes, 2**4, P, L, H, mu, u, CFL)
    for k in range(5, 11):
        X2 = modèle_stochastique(X_ini, temps_final, N_lignes, 2**k, P, L, H, mu, u, CFL)  
        
        erreur = np.abs(X1[num_ligne, 0] - X2[num_ligne, 0])
        for j in range(1, 2**(k-1)):
            if np.abs(X1[num_ligne, j] - X2[num_ligne, 2*j]) > erreur:
                erreur = np.abs(X1[num_ligne, j] - X2[num_ligne, 2*j])
        X1 = X2
        absc.append(L/2**(k-1))
        ords.append(erreur)
    plt.plot(absc, ords)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("Erreur")
    plt.title("Erreur en fonction du pas spatial (échelle log)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()

###régler problème d'affichage de l'animation
def anim_graphe(X_ini, 
                k,  # numéro de la ligne à considérer
                n,   # nombre de frames (= temps final)
                nb_colonnes,
                N_lignes,
           P,
           L, 
           H, 
           mu, 
           u, 
           CFL
                ):
    X = np.copy(X_ini(nb_colonnes))
    fig, ax = plt.subplots()
    delta_x = L/nb_colonnes
    absc = delta_x * (np.arange(nb_colonnes) - nb_colonnes // 2)
    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        nonlocal X
        # évolution d'un pas
        X = modèle_stochastique(X, 1)

        # calcul des ordonnées (décalage périodique)
        ords = [X[k, j] for j in range(nb_colonnes//2 + 1, nb_colonnes)] + [X[k, j] for j in range(0, nb_colonnes//2 + 1)]

        line.set_data(absc, ords)
        return line,
    
    ani = FuncAnimation(fig, update, frames=n, init_func=init,
                        blit=True, interval=100, repeat=False)
    plt.show()


def condition_aux_bords_1(X_ini,
                        temps_final,  
                        N_lignes,
                        nb_colonnes,
                        P,
                        L, 
                        H, 
                        mu, 
                        u, 
                        CFL):
    X = modèle_stochastique(X_ini, temps_final, N_lignes, nb_colonnes, P, L, H, mu, u, CFL)
    diff = 0
    for i in range(N_lignes):
        diff += np.abs(sum(P[i, k]*X[k, 0] for k in range(N_lignes)) - X[i, 0])
    return diff


def graphe_condition_aux_bords_1(condition,
                               X_ini,
                                temps_final,  
                                N_lignes,
                                liste_nb_colonnes,
                                P,
                                L, 
                                H, 
                                mu, 
                                u, 
                                CFL):
    absc=[]
    ords=[]
    for nb_colonnes in liste_nb_colonnes:
        absc.append(L/nb_colonnes)
        ords.append(condition_aux_bords_1(X_ini, temps_final, N_lignes, nb_colonnes, P, L, H, mu, u, CFL))
    plt.plot(absc, ords)
    plt.ylabel(r'$\| P u_n(0) - u_n(0) \|_\infty$')
    plt.title("Etude de la première condition aux bords")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()

def condition_aux_bords_2(X_ini,
                            temps_final,  
                            N_lignes,
                            nb_colonnes,
                            P,
                            L, 
                            H, 
                            mu, 
                            u, 
                            CFL,
                            coefficients=[]):
    if len(coefficients)==0:
        coefficients=[1]*N_lignes
    X = modèle_stochastique(X_ini, temps_final, N_lignes, nb_colonnes, P, L, H, mu, u, CFL)
    def dérivée_discrète_gauche(X, num_ligne):
        return((X[num_ligne, 0] - X[num_ligne, nb_colonnes - 1])/(L/nb_colonnes))
    def dérivée_discrète_droite(X, num_ligne):
        return((X[num_ligne, 1] - X[num_ligne, 0])/(L/nb_colonnes))
    return(np.abs(sum(dérivée_discrète_droite(X, i)*coefficients[i] for i in range(N_lignes)) 
                  - sum(dérivée_discrète_gauche(X, i)*coefficients[i] for i in range(N_lignes))))


def graphe_condition_aux_bords_2(X_ini,
                                temps_final,  
                                N_lignes,
                                liste_nb_colonnes,
                                P,
                                L, 
                                H, 
                                mu, 
                                u, 
                                CFL,
                                coefficients):
    absc=[]
    ords=[]
    for nb_colonnes in liste_nb_colonnes:
        absc.append(L/nb_colonnes)
        ords.append(condition_aux_bords_2(X_ini, temps_final, N_lignes, nb_colonnes, P, L, H, mu, u, CFL, coefficients))
    plt.plot(absc, ords)
    plt.ylabel(r'$\| \sum_{i=1}^N \frac{u_i(x_0) - u_i(x_{n-1})}{\Delta_x} - \sum_{i=1}^N \frac{u_i(x_1) - u_i(x_{0})}{\Delta_x} \|_\infty$')
    plt.title("étude de la deuxième condition aux bords")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()




### cas sans transport : équation de diffusion
def recup_coeffs(f, N_lignes, P, L, N_max, delta_y):
    V = vecteur_propre(P)
    alpha = []
    beta = [[]]*N_lignes
    gamma = [[]]*N_lignes
    for k in range(N_max):
        def g1(x):
            g1 = 0
            for i in range(N_lignes):
                g1 += f(x, i*delta_y)*V[i]*np.cos(2*k*np.pi*x/L)
            return g1
        I, _ = integrate.quad(g1, 0, L)
        alpha.append(2/(V[0]*L)*I)

        for i in range(N_lignes):
            def g2(x):
                return(f(x, i*delta_y)*np.sin(2*k*np.pi*x/L))
            I, _ = integrate.quad(g2, 0, L)
            beta[i].append(2/L*I)

            def g3(x):
                return(f(x, i*delta_y)*np.sin((2*k+1)*np.pi*x/L))
            I, _ = integrate.quad(g3, 0, L)
            gamma[i].append(2/L*I)
    return alpha, beta, gamma


def masse_totale(X):
    return(sum(sum(X)))

def anim_sol(f, #fonction initiale
                    T, #temps_final
                    N_lignes,
                    nb_colonnes,
                    P,
                    L,
                    H,
                    mu,
                    alpha, beta, gamma):
    
    X = bassin_initial(f, N_lignes, nb_colonnes, L, H)
    fig, ax = plt.subplots()
    im = ax.imshow(X, cmap="Greens", interpolation=None, aspect="equal")

    ax.set_xticks(np.arange(-.5, nb_colonnes, 1), minor=True)
    ax.set_yticks(np.arange(-.5, N_lignes, 1), minor=True)
    ax.grid(which='minor', linewidth=1)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    for i in range(N_lignes):
        ax.text(nb_colonnes // 2, i, "R", ha="center", va="center", color="red", fontsize=6)
        

    def update(frame):
        nonlocal X
        # évolution d'un pas
        X = solution_exacte(f, #fonction initiale
                    frame, #temps_final
                    N_lignes,
                    nb_colonnes,
                    P,
                    L,
                    H,
                    mu,
                    alpha, beta, gamma)
        X_show = np.copy(X)
        for i in range(N_lignes):
            for j in range(nb_colonnes):
                j0 = j + nb_colonnes // 2
                if j0 > nb_colonnes - 1:
                    j0 %= nb_colonnes
                X_show[i, j] = X[i, j0]
        im.set_array(X_show)
        return [im]
    
    ani = FuncAnimation(fig, update, frames=200, blit=True, interval=300, repeat=False)
    plt.show()


def f(x, y, N, l, h, p, alpha, beta, gamma):
    V = vecteur_propre(p)
    N_max = len(alpha)
    for i in range(N):
        if y >= i*h/N and y<(i+1)*h/N:
            return(sum(alpha[k]*V[i]*np.cos(2*k*np.pi*x/l)
                   + beta[i][k]*np.sin(2*k*np.pi*x/l)
                   + gamma[i][k]*np.sin((2*k+1)*np.pi*x/l) 
                   for k in range(N_max)))
        
def graphe_temps_final(ligne, f, t, N_lignes, nb_colonnes, p, L, H, mu, u, CFL, alpha, beta, gamma, sol_exacte=True):
    def fonction_initiale(x, y, L):
        return f(x, y, N_lignes, L, H, p, alpha, beta, gamma)
    if sol_exacte:
        X_exa = solution_exacte(fonction_initiale, #fonction initiale
                        t, #temps_final
                        N_lignes,
                        500,
                        p,
                        L,
                        H,
                        mu,
                        alpha, beta, gamma)
    X_approchee = modèle_stochastique(bassin_initial(fonction_initiale, N_lignes, nb_colonnes, L, H), t, N_lignes, nb_colonnes, 
                                    p, L, H, mu, u, CFL)
    if sol_exacte:
        X_exa = np.array(X_exa[ligne - 1]).flatten()
    X_approchee = np.array(X_approchee[ligne - 1]).flatten()

    plt.figure(figsize=(8, 5))
    if sol_exacte:
        plt.plot([k*L/500 for k in range(500)], X_exa, color='blue', label='solution exacte')
    plt.plot([k*L/nb_colonnes for k in range(nb_colonnes)], X_approchee, '.', color='red', label='solution approchée')
    plt.title(f'nombre de colonnes : {nb_colonnes}')
    plt.legend()
    plt.grid(True, which='both')
    plt.show()

def graphe_convergence_a_t_fixe(f, t, N_lignes, p, L, H, mu, u, CFL, alpha, beta, gamma, nb_colonnes_vals, pente=-2):
    erreurs=[]
    def fonction_initiale(x, y, L):
        return f(x, y, N_lignes, L, H, p, alpha, beta, gamma)
    for nb_col in nb_colonnes_vals:
        X_exa = solution_exacte(fonction_initiale, #fonction initiale
                                t, #temps_final
                                N_lignes,
                                nb_col,
                                p,
                                L,
                                H,
                                mu,
                                alpha, beta, gamma)
    
        X_approchee = modèle_stochastique(bassin_initial(fonction_initiale, N_lignes, nb_col, L, H), t, N_lignes, nb_col, 
                                        p, L, H, mu, u, CFL)
        X_exa = np.array(X_exa)
        X_approchee = np.array(X_approchee)
        erreur = np.max(np.abs(X_exa - X_approchee).flatten())
        erreurs.append(erreur)

    plt.figure(figsize=(8, 5))
    plt.loglog(nb_colonnes_vals, erreurs, 'o-', linewidth=2)
    plt.loglog(nb_colonnes_vals, [5*val**pente for val in nb_colonnes_vals], '--', label=f'pente{pente}')
    plt.legend()
    plt.xlabel("nb_colonnes")
    plt.ylabel("‖X_exa - X_approchée‖∞(t) (log)")
    plt.grid(True, which='both')
    plt.show()

#convergence avec erreur max sur toute la durée
def graphe_convergence_norme_inf_en_temps(f, t, N_lignes, p, L, H, mu, u, CFL, alpha, beta, gamma, nb_colonnes_vals, pente=-2):
    erreurs=[]
    def fonction_initiale(x, y, l):
        return f(x, y, N_lignes, L, H, p, alpha, beta, gamma)
    for nb_colonnes in nb_colonnes_vals :
        X_approchee, erreur_max = modèle_stochastique(bassin_initial(fonction_initiale, N_lignes, nb_colonnes, L, H), t, N_lignes, nb_colonnes, 
                                        p, L, H, mu, u, CFL, "advection_diffusion_centrée", True, fonction_initiale, alpha, beta, gamma)
        erreurs.append(erreur_max)

    plt.figure(figsize=(8, 5))
    plt.loglog(nb_colonnes_vals, erreurs, 'o-', linewidth=2)
    plt.loglog(nb_colonnes_vals, [5*val**pente for val in nb_colonnes_vals], '--', label=f'pente{pente}')
    plt.legend()
    plt.xlabel("nb_colonnes")
    plt.ylabel("max {‖X_exa - X_approchée‖∞(t)} 0<t<T (log)")
    plt.grid(True, which='both')
    plt.show()

