## tentative d'implémentation du modèle stochastique du bassin d'algues
# pas de conditions aux bords ni de condition de non accumulation au niveau 0 (dans la roue à aube)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

N = 3 #nombre de couches
L = 50 #longueur du bassin
nb_colonnes_default = 50
delta_x_default = L//nb_colonnes_default
delta_t_default = 1/4
mu = 1
T = 4
M = 40 # nombre d'itérations

#P = np.identity(N)
P = np.matrix(
    [[0, 0, 1],
    [0, 1, 0],
    [1, 0, 0]]
)
#u = [1]*N # liste des vitesses uj
u = [1]*N

def modèle_stochastique(
                        X,  # tableau des valeurs X_i,j
                        temps_final,  
                        N_lignes=N,
                        nb_colonnes=nb_colonnes_default,
                        delta_t=delta_t_default
                        ):
    
    delta_x = L / nb_colonnes
    nu = mu * delta_t / (delta_x * delta_x)
    K = [u[i] * delta_t / (2 * delta_x) for i in range(N_lignes)]
    n = int(temps_final/delta_t)
    X1 = np.copy(X)

    for step in range(n):
        X_tmp = np.copy(X1)

        for i in range(N_lignes):
            for j in range(2, nb_colonnes - 1):# loin de la roue à aube
                X_tmp[i, j] = (
                    (K[i] + nu) * X1[i, j - 1]
                    + (1 - 2 * nu) * X1[i, j]
                    + (nu - K[i]) * X1[i, j + 1]
                )

            X_tmp[i, nb_colonnes - 1] = (# à gauche de la roue
                (1 - 2 * nu) * X1[i, nb_colonnes - 1]
                + sum((nu - K[l]) * P[l, i] * X1[l, 0] for l in range(N_lignes))
                + (K[i] + nu) * X1[i, nb_colonnes - 2]
            )

            X_tmp[i, 0] = (# en la roue
                (K[i] + nu) * X1[i, nb_colonnes - 1]
                + sum((1 - 2 * nu) * P[l, i] * X1[l, 0] for l in range(N_lignes))
                + (nu - K[i]) * X1[i, 1]
            )

            X_tmp[i, 1] = (# à droite
                (1 - 2 * nu) * X1[i, 1]
                + sum((K[l] + nu) * P[l, i] * X1[l, 0] for l in range(N_lignes))
                + (nu - K[i]) * X1[i, 2]
            )

        X1 = X_tmp

    return X1

#il faut à présent appliquer le modèle à quelques exemples que l'on comparera plus tard à l'autre modèle

# On va prendre pour valeurs initiales du tableau X (donc la composition initiale du bassin) celles d'une 
# fonction qu'on aura discrétisée. Par exemple, si l'on prend une fonction affine : f : x, y -> a*x + b*y + c 
# les valeurs des X[i, j] seront celles de la fonction en (i, j)
# prenons pour exemple a = b = c = 1



X_init = np.zeros((N, nb_colonnes_default))
for i in range(N):
    for j in range(nb_colonnes_default):
        X_init[i, j] = i + j + 1

def X_ini(nb_colonnes):
    
    X_ini = np.zeros((N, nb_colonnes))
    for i in range(N):
        for j in range(nb_colonnes):
            X_ini[i, j] = (i + j/nb_colonnes + 1) ## à compléter
    return X_ini
# autre exemple, un polynôme du seconde degré

X_init_1 = np.zeros((N, nb_colonnes_default))
for i in range(N):
    for j in range(nb_colonnes_default):
        X_init_1[i, j] = i*i + j*j + i*j + i + j + 1

# une version encore plus contrastée

X_init_2 = np.zeros((N, nb_colonnes_default))
for i in range(N):
    for j in range(nb_colonnes_default):
        X_init_2[i, j] = np.exp(i - 25) * np.exp(j - 25)

# un bloc d'algues au milieu :

X_init_3 = np.zeros((N, nb_colonnes_default))
for i in range(N//2, N//2 + 1):
    for j in range(nb_colonnes_default//2, nb_colonnes_default//2 + 1):
        X_init_3[i, j] = 100

#un cercle d'algues

X_init_4 = np.zeros((N, nb_colonnes_default))
for i in range(N):
    for j in range(nb_colonnes_default):
        if (i - 5)*(i - 5) + (j - 5)*(j - 5) <= 4:
            X_init_4[i, j] = 100

#  pour mieux visualiser, implémentons un algorithme qui renvoie un dessin du bassin à partir de la valeur de X

def dessin_bassin(X):
    fig, ax = plt.subplots()
    im = ax.imshow(X, cmap ="Greens", interpolation = None)
    ax.set_xticks(np.arange(-.5, X.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, X.shape[0], 1), minor=True)
    ax.grid(which='minor', linewidth=1)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.show()

def anim(nb_colonnes=nb_colonnes_default):
    X_anim = np.copy(X_ini(nb_colonnes))

    fig, ax = plt.subplots()
    im = ax.imshow(X_anim, cmap="Greens", interpolation=None, aspect="equal")

    # grille comme dans dessin_bassin
    ax.set_xticks(np.arange(-.5, nb_colonnes, 1), minor=True)
    ax.set_yticks(np.arange(-.5, N, 1), minor=True)
    ax.grid(which='minor', linewidth=1)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    def update(frame):
        global X_anim
        X_anim = modèle_stochastique(X_anim, 1, N, nb_colonnes)   # une itération comme dans la boucle
        X_show = np.copy(X_anim)
        for i in range(N): # autre ordre des cases pour placer la roue au milieu du graphe
            for j in range(nb_colonnes):
                j0 = j + nb_colonnes//2
                if j0 > nb_colonnes - 1:
                    j0 %= nb_colonnes
                X_show[i, j] = X_anim[i, j0]
            ax.text(nb_colonnes//2, i, "R", ha="center", va="center", color="red", fontsize=4)
        im.set_array(X_show)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=300, interval=100, blit=False)
    plt.show()

# On peut bien évidemment tester le modèle avec différentes matrices de mélange, différentes vitesse en fonction des couches 
#l'idée est d'étudier la convergence du modèle quand les delta_x et delta_t tendent simultanément vers 0 ...

##faire un graphe de la fonction au temps T pour voir son évolution en fonction du pas de temps et d'espace

def graphe(X_ini, 
           k, #numéro de la ligne à considérer
           temps_final=T, #instant final
           nb_colonnes=nb_colonnes_default
           ):
    #pour respecter la conditionn CFL :
    delta_t = L*L/(4*mu*nb_colonnes*nb_colonnes)
    X = modèle_stochastique(X_ini(nb_colonnes), temps_final, N, nb_colonnes, delta_t)
    absc = [delta_x_default*(j - nb_colonnes//2) for j in range(nb_colonnes)]
    ord = [X[k, j] for j in range(nb_colonnes//2 + 1, nb_colonnes)] + [X[k, j] for j in range(0, nb_colonnes//2 + 1)]
    plt.plot(absc, ord, 'o')
    plt.show()

graphe(X_ini, 0, 10, 100)
graphe(X_ini, 0, 10, 200)
graphe(X_ini, 0, 10, 400)
graphe(X_ini, 0, 10, 800)
graphe(X_ini, 1, 10, 100)
graphe(X_ini, 2, 10, 100)




##étude du cas N=1, ordre de convergence du schéma (on ne calcule pas un sup sur tous les n de la norme de l'erreur, 
# mais seulement sur l'instant final n).

def erreur(k,
           n,
           J # = nb de colonnes, qui va donner la valeur de delta_x
            ):#on prend par défaut la ligne 0 du bassin et l'instant 30 (arbitraire)
    # mais delta_t dépend de nb_colonnes donc il faut que n soit proportionnel à nb_colonnes*nb_colonnes pour rester 
    # au même instant (car delta_t = L*L/(4*mu*nb_colonnes*nb_colonnes))
    X1 = modèle_stochastique(X_ini(J), n*J*J, N, J)
    X2 = modèle_stochastique(X_ini(J+1), n*(J+1)*(J+1), N, J+1)
    erreur = np.abs(X1[k, 0] - X2[k, 0])
    for j in range(1, J):
        if np.abs(X1[k, j] - X2[k, j]) > erreur:
            erreur = np.abs(X1[k, j] - X2[k, j])
    
    return(erreur)

def graphe_log_erreur(n): 
    absc= []
    ords = []
    for J in range(10, 20):
        absc.append(L/J)
        ords.append(erreur(0, n, J))
    plt.plot(absc, ords)
    plt.yscale('log')
    plt.ylabel("Erreur")
    plt.title("Erreur en fonction du pas spatial (échelle log)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()


graphe_log_erreur(30)

###régler problème d'affichage de l'animation
def anim_graphe(X_ini, 
                k,  # numéro de la ligne à considérer
                n,   # nombre de frames (= temps final)
                nb_colonnes=nb_colonnes_default
                ):
    X = np.copy(X_ini)
    fig, ax = plt.subplots()

    absc = delta_x_default * (np.arange(nb_colonnes) - nb_colonnes // 2)
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

#anim_graphe(X_ini, 0, 400)