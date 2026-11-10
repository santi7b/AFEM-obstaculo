import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
import time


def construir_aristas(nodos, elementos):
    
    #Se definen las aristas de los elementos de la malla, se clasifican las aristas del borde.
    
    N_elem = len(elementos) #creamos las aristas pero están repetidas (crudas).
    raw_edges = np.vstack([
        elementos[:, [1, 2]], # Arista opuesta a v0
        elementos[:, [2, 0]], # Arista opuesta a v1
        elementos[:, [0, 1]]  # Arista opuesta a v2
    ])
    sorted_edges = np.sort(raw_edges, axis=1) #Ordena los dos índices de cada arista para que el nodo de menor índice quede siempre primero
    unique_edges, inverse_indices = np.unique(sorted_edges, axis=0, return_inverse=True) 
    # unique_edges: Elimina las aristas repetidas y devuelve la lista global única de aristas de la malla. La fila i contiene los dos nodos extremos de la arista global i
    # inverse_indices: Es un vector que indica, para cada arista de raw_edges, qué índice le corresponde en la lista ordenada unique_edges
    elem2edges = inverse_indices.reshape(3, N_elem).T #Reorganiza inverse_indices en una matriz, ahora elem2edges[k] contiene los 3 números de arista global que forman el triángulo k
    
    # Coordenadas de los puntos medios de las aristas (grados de libertad de Crouzeix-Raviart)
    edge_midpoints = 0.5 * (nodos[unique_edges[:, 0]] + nodos[unique_edges[:, 1]])
    
    # Identificar aristas de frontera Dirichlet
    #Evalúa si el punto medio de la arista está sobre las paredes del dominio. Retorna un vector booleano True/False de tamaño N_edges
    # Identificar aristas de frontera Dirichlet: el borde es la caja que encierra la malla
    # (el refinamiento nunca cambia esta caja, asi que sirve para cualquier dominio rectangular).
    eps = 1e-12
    xmin, xmax = nodos[:, 0].min(), nodos[:, 0].max()
    ymin, ymax = nodos[:, 1].min(), nodos[:, 1].max()
    is_boundary = (
        (np.abs(edge_midpoints[:, 0] - xmin) < eps) |
        (np.abs(edge_midpoints[:, 0] - xmax) < eps) |
        (np.abs(edge_midpoints[:, 1] - ymin) < eps) |
        (np.abs(edge_midpoints[:, 1] - ymax) < eps)
    )

    return unique_edges, elem2edges, edge_midpoints, is_boundary


def crear_malla_unidad(N,a):
    
    #Crea una malla uniforme de triángulos en [-1, 1]^2 con N divisiones por lado (N+1 puntos por lado).
    

    x = np.linspace(-a, a, N + 1)
    y = np.linspace(-a, a, N + 1)
    X, Y = np.meshgrid(x, y)
    nodos = np.column_stack([X.ravel(), Y.ravel()])
    
    elementos = []
    for j in range(N): #indexado de vértices en la malla: por filas de abajo hacia arriba. Y creacion de los elementos (tríangulos) dividiendo subcuadrados en dos triangulos por su diagonal.
        for i in range(N): 
            n0 = j * (N + 1) + i
            n1 = n0 + 1
            n2 = n0 + (N + 1)
            n3 = n2 + 1
            elementos.append([n0, n1, n3])
            elementos.append([n0, n3, n2])
    elementos = np.array(elementos)

    # etiquetar la arista de referencia de cada triangulo (necesario para la biseccion) y recalcular aristas
    elementos = preparar_arista_ref(nodos, elementos)
    
    # Crea las aristas
    unique_edges, elem2edges, edge_midpoints, is_boundary = construir_aristas(nodos, elementos)
    
    return nodos, elementos, unique_edges, elem2edges, edge_midpoints, is_boundary



def ensamblar_matrices_cr(nodes, elements, elem2edges, edge_midpoints, f_func, chi_func, kappa):
    
    #Ensambla la matriz de rigidez R, masa diagonal M_diag, carga F y obstaculo Chi.
    
    N_edges = len(edge_midpoints)
    N_elem = len(elements)
    
    I_list, J_list, V_list = [], [], []
    
    F = np.zeros(N_elem)
    

    
       

    centros= np.zeros((N_elem, 2))
    grad_phi_all = np.zeros((N_elem, 3, 2))
    areas = np.zeros(N_elem)
    
    for k in range(N_elem):
        elem_v = elements[k]
        pts = nodes[elem_v]
        x0, y0 = pts[0]
        x1, y1 = pts[1]
        x2, y2 = pts[2]
        
        # Área del triángulo
        area = 0.5 * abs((x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0))
        areas[k] = area
        
        # Gradientes en coordenadas locales
        grad_phi = np.zeros((3, 2))
        grad_phi[0] = [y2 - y1, x1 - x2] / area
        grad_phi[1] = [y0 - y2, x2 - x0] / area
        grad_phi[2] = [y1 - y0, x0 - x1] /  area

          
        centros[k] = np.mean(pts, axis=0) # Baricentro
        grad_phi_all[k] = grad_phi          # Matriz (3, 2) con los gradientes en el lugar k
        
        
        S_loc = kappa * area * (grad_phi @ grad_phi.T)

        # f_h = Pi_h f, es constante en el triángulo
        F[k] =  f_func(centros[k][0], centros[k][1])

        
        
        
        e_indices = elem2edges[k]
        for i in range(3):
            e_i = e_indices[i]

                     
                       
                        
            for j in range(3):
                e_j = e_indices[j]
                I_list.append(e_i)
                J_list.append(e_j)
                V_list.append(S_loc[i, j])
                
    S = sp.coo_matrix((V_list, (I_list, J_list)), shape=(N_edges, N_edges)).tocsr()

    # ensamblar la matriz P para newton
    rows = elem2edges.ravel()                 # 3 aristas por elemento
    cols = np.repeat(np.arange(N_elem), 3)
    vals = np.repeat(areas / 3.0, 3)
    P = sp.csc_matrix((vals, (rows, cols)), shape=(N_edges, N_elem))

    Chi = chi_func(centros[:, 0], centros[:, 1])

    # Obstáculo evaluado en los vértices/nodos (para la solución conforme P1)
    Chi_nodes = chi_func(nodes[:, 0], nodes[:, 1])

    Chi_edges = chi_func(edge_midpoints[:, 0], edge_midpoints[:, 1])
    

    
    
    return S, P, F, Chi, Chi_nodes, Chi_edges, centros, areas , grad_phi_all

def solve(S, P, F,  Chi_free, c):

    # Etapa (1): Aplica el método de Newton semi-suave para tener las aproximaciones u_h, \lambda_h

    stop= 1e-10
    N_elem=len(F)
    N_edges = len(Chi_free)
    
    U = np.maximum(0.0,Chi_free)
    X = Chi_free

    #multiplicador de Lagarange inicializdo como cero
    L = np.zeros(N_elem) #  el multiplicador por elemento es cero en la parte inactiva.


    historial_activo=[]

    norma = 1.0
    iteracion = 0
    while norma > stop and iteracion < 200:
        iteracion += 1

        mascara_activa = (L + c * P.T @ (U - X)) < 0

        A = np.where(mascara_activa)[0] #indices activos
        

        historial_activo.append(len(A))

        U_new = np.zeros(N_edges)

        
        

        L_new = np.zeros(N_elem) # lambda = 0 en los triángulos inactivos

        if len(A) == 0: #si no hay triángulos activos es solo el problema sin restricción
            U_new = spsolve(S.tocsc(),  P @ F)
        else:

            B = P[:, A] #reducimos el sistema a ([S,B],[B^t,0)*(U,L[A]) =(B F[A], B^t X)

            K = sp.bmat([[S, B], [B.T, None]], format='csc')
            rhs = np.concatenate([P @ F, B.T @ X])

            sol = spsolve(K, rhs)
            


            U_new = sol[:N_edges]
            L_new[A] = sol[N_edges:]

            

        norma = np.max(np.abs(U_new-U))

        U = U_new
        L = L_new 

    info = {
    'n_iter': iteracion,
    'historial_activo': historial_activo
    }

    

    return U, L, info

def reconstruir_flujo_dual(elements, elem2edges, grad_phi_all,  U_cr, L, F):

    #Etapa (2): Reconstrucción explícita del flujo dual z_h \in RT_0(\Omega), mediante la fórmula de Marini.

    N_elem = len(elements)

    z_T = np.zeros((N_elem, 2))
    c_div = np.zeros(N_elem)
    
    

    for k in range(N_elem):
        
        e_indices = elem2edges[k]
        u_loc = U_cr[e_indices]  # vector de coordenadas locales de u_h

        grad_u_T = (u_loc @ grad_phi_all[k]) 
        z_T[k] = grad_u_T

        
        
        
        c_div[k] = 0.5 * (L[k] - F[k] ) #término de divergencia del flujo

    return z_T, c_div

def reconstruir_solucion_conforme(nodes, elements, elem2edges, U_cr, Chi_nodes, dirichlet_nodes):
    
    #Etapa (3): Promedio en nodos para construir \bar{u}_h \in H^1_0 admisible (\bar{u}_h >= \chi).
    
    N_nodes = len(nodes)
    N_elem = len(elements)
    
    u_sum = np.zeros(N_nodes)
    node_count = np.zeros(N_nodes, dtype=int) # registra cuantos triángulos aportan para el promedio de cada vértice

    # 1. Acumular extrapolaciones desde cada triángulo hacia sus vértices
    for k in range(N_elem):
        verts = elements[k]      # [v0, v1, v2]
        edges = elem2edges[k]    # [e0, e1, e2] (opuestas a v0, v1, v2)
        u_e = U_cr[edges]

         # Valor de u_h^cr evaluado en cada uno de los 3 vértices del triángulo
        u_vert = np.array([
            u_e[1] + u_e[2] - u_e[0],  # Vértice 0
            u_e[0] + u_e[2] - u_e[1],  # Vértice 1
            u_e[0] + u_e[1] - u_e[2]   # Vértice 2
        ])
        
        for i in range(3):
            u_sum[verts[i]] += u_vert[i]
            node_count[verts[i]] += 1

    # 2. Promedio en nodos
    u_conforme = u_sum / node_count
    
    # 3. Garantizar admisibilidad primal (\bar{u}_h >= \chi)
    u_conforme = np.maximum(u_conforme, Chi_nodes)
    
    # 4. Imponer condición Dirichlet cero en la frontera
    u_conforme[dirichlet_nodes] = 0.0
    
    return u_conforme

def calcular_estimador_gap(nodes, elements, areas, centros, grad_phi_all, u_conforme, z_const, c_div, lambda_elem, Chi_h):
    
    #Etapa (4): Cálculo de la Brecha Primal-Dual \eta_gap^2. Otorga un control de error exacto garantizado sin constantes desconocidas.
    
    N_elem = len(elements)
    
    eta_flux_elem = np.zeros(N_elem)
    eta_obs_elem = np.zeros(N_elem)
    
    for k in range(N_elem):
        pts = nodes[elements[k]]
        p0, p1, p2 = pts
        area = areas[k]
        x_T = centros[k]
        
        # 1. Gradiente de la función P1 conforme \nabla \bar{u}_h en T
        
    
        
        u_nodes_loc = u_conforme[elements[k]]
        grad_u_bar =  -0.5 * (u_nodes_loc @ grad_phi_all[k])  # Vector 2D 
        
        # 2. Término de error de Flujo \int_T |\nabla \bar{u}_h - z_h^rt|^2 dx
        diff_const = grad_u_bar - z_const[k]
        term1 = area * np.dot(diff_const, diff_const)
        
        # integral \int_T |x-x_t|^2 dx
        d01 = np.sum((p0 - p1)**2)
        d12 = np.sum((p1 - p2)**2)
        d20 = np.sum((p2 - p0)**2)
        inertia = (area / 36.0) * (d01 + d12 + d20)
        
        term2 = (c_div[k]**2) * inertia
        eta_flux_elem[k] = 0.5 * (term1 + term2)
        
        # 3. Término de error de Obstáculo (Complementariedad)
        u_bar_centro = np.mean(u_nodes_loc)
        gap_obs = u_bar_centro - Chi_h[k]
        
        # \lambda_T <= 0, por lo que -2 * \lambda_T * area * gap_obs >= 0
        eta_obs_elem[k] = - lambda_elem[k] * area * max(0.0, gap_obs)

    eta_gap_elem = eta_flux_elem + eta_obs_elem #\eta_{gap}^2 en cada elemento.
        
    # Estimadores globales
    eta_flux2 = np.sum(eta_flux_elem)
    eta_obs2 = np.sum(eta_obs_elem)
    eta_gap_global = eta_flux2 + eta_obs2
    
    
    
    return  eta_gap_elem, eta_gap_global

def marcar_elementos(eta_gap_elem):

    
    #Obtiene el subconjunto mínimo de elementos que acumulan el mayor error. Utilizamos la estrategia Dörfler de búsqueda por bisección

    
    theta2 = 0.25

    index_gap = np.argsort(eta_gap_elem)[:: -1] #obtiene los índices de los elementos ordenados de mayor a menor (cambiamos mayor a menor con -> [::-1])

    suma_acumulada = np.cumsum(eta_gap_elem[index_gap]) #array con la suma acumulada de los gap´s ordenados.

    indice = np.searchsorted(suma_acumulada, theta2 * suma_acumulada[-1], side='left') #busca en la suma acumulada donde ebería colocar el valor  theta2 * eta_gap_global

    M_indices = index_gap[ : indice + 1]

    return M_indices


def _key(a, b):
    _BASE = np.int64(1) << np.int64(31)
    a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
    return np.minimum(a, b) * _BASE + np.maximum(a, b)

def nodos_frontera(nodes, tol=1e-12):
    """Nodos sobre el borde del rectangulo que encierra la malla (sirve para cualquier lado)."""
    xmin, xmax = nodes[:, 0].min(), nodes[:, 0].max()
    ymin, ymax = nodes[:, 1].min(), nodes[:, 1].max()
    return np.where((np.abs(nodes[:, 0] - xmin) < tol) | (np.abs(nodes[:, 0] - xmax) < tol) |
                    (np.abs(nodes[:, 1] - ymin) < tol) | (np.abs(nodes[:, 1] - ymax) < tol))[0]

def preparar_arista_ref(nodes, elements):
    """
    Reordena cada triangulo como [n, a, b] (orden antihorario conservado) con (a,b) = arista mas larga
    (arista de referencia) y n el vertice opuesto. Llamar UNA vez sobre la malla inicial y
    recalcular luego construir_aristas. (refinar_malla_rgb usa esa arista para cerrar la malla sin degradar los angulos)
    """
    P = nodes[elements]
    L = np.stack([np.linalg.norm(P[:, 1] - P[:, 2], axis=1),
                  np.linalg.norm(P[:, 2] - P[:, 0], axis=1),
                  np.linalg.norm(P[:, 0] - P[:, 1], axis=1)], axis=1)
    i = np.argmax(np.round(L / L.max(), 10), axis=1)
    rot = (i[:, None] + np.arange(3)) % 3
    return elements[np.arange(len(elements))[:, None], rot].astype(np.int64)

def refinar_malla_rgb(nodes, elements, marked_elements, dirichlet_nodes=None):
    """
    Refinamiento adaptativo conforme Red-Green-Blue (RGB) con arista de referencia, vectorizado.
    Cada triangulo es [n, a, b]: (a,b) es su arista de referencia (la mas larga, ver preparar_arista_ref) y n el vertice opuesto.
      - los elementos marcados se refinan en rojo: se marcan sus 3 aristas.
      - clausura (conformidad): si un triangulo tiene alguna arista marcada, tambien se marca su arista de referencia
        (se repite hasta que no cambie nada). Asi se bisecan primero las aristas largas y los angulos no se degradan.
      - cada triangulo se divide segun cuantas aristas tiene marcadas: solo la de referencia -> verde (2 hijos),
        la de referencia y otra -> azul (3 hijos), las 3 -> rojo (4 hijos).
    La malla de entrada debe haberse preparado con preparar_arista_ref y los hijos conservan la convencion [n, a, b].

    Parametros:
      - nodes: Array (N_nodes, 2) con las coordenadas de los vertices.
      - elements: Array (N_elem, 3) con los indices de los nodos de cada triangulo.
      - marked_elements: Lista o array con los indices M_k de elementos marcados para refinamiento.
      - dirichlet_nodes: no se usa, se deja para no cambiar la llamada (la frontera se recalcula geometricamente).

    Retorna:
      - nodes_new, elements_new, dirichlet_new (nodos del borde de la nueva malla).
    """
    marked_elements = np.asarray(marked_elements, dtype=np.int64)
    el = elements.astype(np.int64)
    if marked_elements.size == 0:
        return nodes.copy(), el.copy(), nodos_frontera(nodes)

    _BASE = np.int64(1) << np.int64(31)
    n_, a_, b_ = el[:, 0], el[:, 1], el[:, 2]

    # claves de las 3 aristas de cada triangulo: R=(a,b) referencia, L=(n,a), Q=(b,n)
    kR = _key(a_, b_)
    kL = _key(n_, a_)
    kQ = _key(b_, n_)

    # PASO 1: marcar las 3 aristas de los elementos marcados (refinamiento rojo)
    mk = np.unique(np.concatenate([kR[marked_elements], kL[marked_elements], kQ[marked_elements]]))

    # PASO 2: clausura. Si un triangulo tiene una arista marcada, su arista de referencia tambien debe estarlo.
    while True:
        mR, mL, mQ = np.isin(kR, mk), np.isin(kL, mk), np.isin(kQ, mk)
        need = (mL | mQ) & ~mR
        if not need.any():
            break
        mk = np.union1d(mk, kR[need])

    # PASO 3: un nodo nuevo en el punto medio de cada arista marcada (mk esta ordenado y sin repetidos)
    lo, hi = mk // _BASE, mk % _BASE
    ids = len(nodes) + np.arange(mk.size)
    nodes_new = np.vstack([nodes, 0.5 * (nodes[lo] + nodes[hi])])

    def medio(k, marcada):
        out = np.full(len(k), -1, dtype=np.int64)
        out[marcada] = ids[np.searchsorted(mk, k[marcada])]
        return out
    m = medio(kR, mR)    # punto medio de la arista de referencia (a,b)
    ml = medio(kL, mL)   # punto medio de (n,a)
    mq = medio(kQ, mQ)   # punto medio de (b,n)

    def tri(mask, x, y, z):
        return np.column_stack([x[mask], y[mask], z[mask]])

    bloques = []

    # CASO 0: elemento no modificado
    s = ~mR
    bloques.append(el[s])

    # CASO GREEN (verde): solo la arista de referencia -> 2 hijos, [m,n,a] y [m,b,n] (arista de referencia = opuesta al nodo nuevo m)
    s = mR & ~mL & ~mQ
    bloques += [tri(s, m, n_, a_), tri(s, m, b_, n_)]

    # CASO BLUE (azul): referencia + otra arista -> 3 hijos (se biseca ademas el hijo que contiene a la otra arista)
    s = mR & mL & ~mQ          # referencia y (n,a)
    bloques += [tri(s, ml, m, n_), tri(s, ml, a_, m), tri(s, m, b_, n_)]
    s = mR & ~mL & mQ          # referencia y (b,n)
    bloques += [tri(s, m, n_, a_), tri(s, mq, m, b_), tri(s, mq, n_, m)]

    # CASO RED (rojo): las 3 aristas -> 4 hijos (los 3 de las esquinas y el central son semejantes al padre,
    # y conservan el mismo orden [n,a,b], asi que la arista de referencia de cada hijo es la analoga a la del padre)
    s = mR & mL & mQ
    bloques += [tri(s, ml, a_, m), tri(s, mq, m, b_), tri(s, n_, ml, mq), tri(s, m, mq, ml)]

    elements_new = np.vstack(bloques).astype(np.int64)

    # la frontera se recalcula geometricamente (no por herencia entre extremos), asi que una diagonal interior
    # que une dos nodos del borde no genera un nodo de frontera falso
    return nodes_new, elements_new, nodos_frontera(nodes_new)




def graficar_cuatro_estados(capturas):
    if not capturas:
        raise ValueError("No hay estados guardados para graficar.")

    # Seleccionar cuatro estados distribuidos entre el inicial y el final.
    indices = np.rint(
        np.linspace(0, len(capturas) - 1, min(4, len(capturas)))
    ).astype(int)

    estados = [capturas[i] for i in np.unique(indices)]
    n_estados = len(estados)

    fig = plt.figure(figsize=(5 * n_estados, 12),
                     constrained_layout=True)

    for j, estado in enumerate(estados):
        nodes = estado["nodes"]
        elements = estado["elements"]
        u = estado["u"]
        eta = estado["eta"]

        triang = mtri.Triangulation(
            nodes[:, 0], nodes[:, 1], elements
        )

        titulo = (
            f"Iteración {estado['iteracion']}\n"
            f"Elementos: {len(elements)}"
        )

        # 1. Solución u_h en 3D
        ax = fig.add_subplot(3, n_estados, j + 1, projection="3d")
        superficie = ax.plot_trisurf(
            triang, u,
            cmap="viridis",
            linewidth=0.08,
            antialiased=True
        )
        ax.set_title(titulo)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(r"$\bar{u}_h$")
        ax.view_init(elev=28, azim=-55)
        fig.colorbar(superficie, ax=ax, shrink=0.65, pad=0.08)

        # 2. Distribución espacial del gap por elemento
        ax = fig.add_subplot(3, n_estados, n_estados + j + 1)

        eta_pos = np.maximum(eta, 1e-16)
        vmin = max(float(np.min(eta_pos)), 1e-16)
        vmax = max(float(np.max(eta_pos)), vmin * 1.001)

        mapa = ax.tripcolor(
            triang,
            facecolors=eta_pos,
            shading="flat",
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax)
        )

        ax.set_title(
            r"$\eta_{\mathrm{gap},T}^2$" + "\n"
            + f"Global: {estado['eta_global']:.3e}"
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(mapa, ax=ax, shrink=0.82)

        # 3. Malla triangular
        ax = fig.add_subplot(3, n_estados, 2 * n_estados + j + 1)
        ax.triplot(triang, color="black", linewidth=0.35)
        ax.set_title("Malla adaptativa\n" + titulo)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(
        "Problema del obstáculo: solución, gap y refinamiento adaptativo",
        fontsize=15
    )

    fig.savefig(
        "graficos_obstaculo_cuatro_estados.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.show()

def graficar_solucion_final(capturas):
    """Una única figura para la solución de la última captura."""
    if not capturas:
        raise ValueError("No hay capturas para graficar.")

    estado = capturas[-1]
    nodes = estado["nodes"]
    elements = estado["elements"]
    u = estado["u"]

    triang = mtri.Triangulation(
        nodes[:, 0], nodes[:, 1], triangles=elements
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    superficie = ax.plot_trisurf(
        triang,
        u,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.03,
        antialiased=True
    )

    ax.set_title(
        f"Solución final $\\bar{{u}}_h$ — iteración {estado['iteracion']}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(r"$\bar{u}_h$")
    fig.colorbar(superficie, ax=ax, shrink=0.7, pad=0.1,
                 label=r"$\bar{u}_h$")
    ax.view_init(elev=30, azim=-60)
    plt.tight_layout()


def graficar_gap_y_malla(capturas):
    """Otra figura: gap y malla de las capturas seleccionadas."""
    if not capturas:
        raise ValueError("No hay capturas para graficar.")

    indices = np.unique(np.rint(
        np.linspace(0, len(capturas) - 1, min(4, len(capturas)))
    ).astype(int))
    estados = [capturas[i] for i in indices]
    n = len(estados)

    fig, axes = plt.subplots(
        2, n, figsize=(5 * n, 9), squeeze=False
    )

    for j, estado in enumerate(estados):
        nodes = estado["nodes"]
        elements = estado["elements"]
        eta = np.maximum(np.asarray(estado["eta"]), 0.0)

        triang = mtri.Triangulation(
            nodes[:, 0], nodes[:, 1], triangles=elements
        )

        # Gap local por elemento, representado en el centroide.
        centros = nodes[elements].mean(axis=1)
        sc = axes[0, j].scatter(
            centros[:, 0],
            centros[:, 1],
            c=np.log10(np.maximum(eta, 1e-16)),
            cmap="magma",
            s=12
        )
        axes[0, j].set_title(
            f"Gap — iteración {estado['iteracion']}\n"
            f"Gap global: {estado['eta_global']:.3e}"
        )
        axes[0, j].set_xlabel("x")
        axes[0, j].set_ylabel("y")
        axes[0, j].set_aspect("equal")
        fig.colorbar(sc, ax=axes[0, j], label=r"$\log_{10}(\mathrm{gap})$")

        # Malla triangular.
        axes[1, j].triplot(
            triang, color="black", linewidth=0.45
        )
        axes[1, j].set_title(
            f"Malla — iteración {estado['iteracion']}\n"
            f"Elementos: {len(elements)}"
        )
        axes[1, j].set_xlabel("x")
        axes[1, j].set_ylabel("y")
        axes[1, j].set_aspect("equal")

    fig.suptitle("Estimador de gap y refinamiento de la malla", y=1.02)
    plt.tight_layout()




# --- Definición de funciones del problema ---
def f_func(x, y):
    return -2.0 *np.ones_like(x)



   
# Definición de funciones del problema 
OBSTACULO = 4 # 0: (sin obstáculo) | 1: domo cónico | 2: domo suave | 3: dos domos | 4: pirámide | 5: anillo de radio | 6: cresta | 7: domo elíptico | 8: domo con base plana |9: cinco domos.

def chi_func(x, y):
    r = np.sqrt(x**2 + y**2)
    if OBSTACULO == 0:    # ejemplo del artículo: chi = 0
        return -5.2*np.ones_like(x)
    if OBSTACULO == 1:    # domo cónico central (no suave en r = 0 y en r = 0.6)
        return np.maximum(0.0, 0.6 - r)
    if OBSTACULO == 2:    # domo suave (C^1), radio 0.8 y altura 0.8
        return 0.8 * np.maximum(0.0, 1.0 - (r / 0.8)**2)**2
    if OBSTACULO == 3:    # dos domos cónicos de distinta altura, centrados en (-0.7, 0) y (0.7, 0)
        r1 = np.sqrt((x + 0.7)**2 + y**2)
        r2 = np.sqrt((x - 0.7)**2 + y**2)
        return np.maximum(0.0, np.maximum(0.7 - 1.4 * r1, 0.5 - 1.0 * r2))
    if OBSTACULO == 4:    # pirámide
        return np.maximum(0.0, 0.6 - np.maximum(np.abs(x), np.abs(y)))
    if OBSTACULO == 5:    # anillo de radio 0.8
        return np.maximum(0.0, 0.5 - 2.5 * np.abs(r - 0.8))
    if OBSTACULO == 6:    # pared (cresta) a lo largo del eje x
        return np.maximum(0.0, np.minimum(0.5 - 2.5 * np.abs(y), 3.0 * (1.2 - np.abs(x))))
    if OBSTACULO == 7:    # domo elíptico suave, corrido del centro
        q = ((x - 0.4) / 0.9)**2 + ((y + 0.3) / 0.5)**2
        return 0.7 * np.maximum(0.0, 1.0 - q)**2
    if OBSTACULO == 8:    #domo con base plana.
        return np.where(r <= 0.5, 2.0 * np.ones_like(x), np.maximum( 0.0, 3.0 - 2*r))
    if OBSTACULO == 9:    # cinco domos (uno central y cuatro en (+-1, +-1)), vale 0 en el borde
        return np.maximum(0.0, 0.5 * np.cos(np.pi * x*1.5/a) * np.cos(np.pi * y*1.5/a))
    

print("=== INICIANDO...")
t_inicial=time.perf_counter()

# Inicialización de Malla y Datos 
N = 32
stop = 1e-2 #epsilon de parada.
gap = 1.0
kappa= 1.0 #constante física.
a = 2 #tamaño del dominio

#se crea la malla
nodes, elements, edges, elem2edges, edge_midpoints, is_boundary = crear_malla_unidad(N,a)


# nodos en la frontera
eps= 1e-12
dirichlet_nodes = nodos_frontera(nodes)  

historial_eta = []
historial_dof = []
iteracion = 0

capturas = []

while gap > stop:
    iteracion += 1
    
    #Etapa Solve

    # Ensamble de las matrices
    S, P, F, Chi, Chi_nodes, Chi_edges, centros, areas, grad_phi_all = ensamblar_matrices_cr(nodes, elements, elem2edges, edge_midpoints, f_func, chi_func,kappa)

    # Filtrar solo grados de libertad interiores (nodos libres)
    indices_libres = np.where(~is_boundary)[0]
    N_free = len(indices_libres)

    


    S_free = S[indices_libres, :][:, indices_libres]
    P_free = P[indices_libres].tocsc()
    Chi_free = Chi_edges[indices_libres]
    
    

    c= 1.0

    #Se resuelve el sistema en los grados de libertad
    U_free, L, info = solve(S_free, P_free, F, Chi_free, c)

    #Se impone la condicion de frontera
    U_cr = np.zeros(len(edge_midpoints)) # (cambio) volvemos a poner los grados de libertad de frontera (valen 0)
    U_cr [indices_libres] = U_free

    z_T, c_div = reconstruir_flujo_dual(elements, elem2edges, grad_phi_all,  U_cr, L, F)

    #reconstruir u_h conforme
    u_h = reconstruir_solucion_conforme(nodes, elements, elem2edges, U_cr, Chi_nodes, dirichlet_nodes)

    #calcula el gap primal-dual
    Chi_h = Chi_edges[elem2edges].mean(axis=1)# promedio del obstáculo en los puntos medios de las aristas.
    eta_gap_elem, eta_gap_global = calcular_estimador_gap(nodes, elements, areas, centros, grad_phi_all, u_h, z_T, c_div, L, Chi_h)

    capturas.append({
    "iteracion": iteracion,
    "nodes": nodes.copy(),
    "elements": elements.copy(),
    "u": u_h.copy(),
    "eta": eta_gap_elem.copy(),
    "eta_global": float(eta_gap_global)
    })

    N_dof = N_free + len(elements) # grados de libertad como en el artículo: dim(S^{1,cr}_D) + dim(Pi_h(S^{1,cr}_D))
    historial_eta.append(eta_gap_global)
    historial_dof.append(N_dof)
    print(f"it={iteracion:2d}  elementos={len(elements):6d}  dof={N_dof:6d}  iter_newton={info['n_iter']:2d}  activos={info['historial_activo'][-1]:5d}  eta_gap^2={eta_gap_global:.4e}")

    gap = eta_gap_global

    if gap > stop:

        #marcar y refinar.
        M_indices = marcar_elementos(eta_gap_elem)
        nodes, elements, dirichlet_nodes = refinar_malla_rgb(nodes, elements, M_indices, dirichlet_nodes)

        # reconstruir las aristas
        edges, elem2edges, edge_midpoints, is_boundary = construir_aristas(nodes, elements)
    if iteracion >= 60:
        break

t_final=time.perf_counter()-t_inicial
print(t_final, "s")


#graficar_cuatro_estados_interactivo(capturas)

#graficar_cuatro_estados(capturas)

#graficar_solucion_final(nodes, elements, u_h)
print("=== Graficando...")   

graficar_gap_y_malla(capturas)



plt.figure()
plt.loglog(historial_dof, historial_eta, 'o-', label=r'$\eta_{gap}^2$')
plt.loglog(historial_dof, historial_eta[0] * (np.array(historial_dof) / historial_dof[0])**(-1.0), 'k--', label='$N^{-1}$')
plt.xlabel('grados de libertad')
plt.legend()

graficar_solucion_final(capturas)

plt.show()

print("=== Finalizado.")
