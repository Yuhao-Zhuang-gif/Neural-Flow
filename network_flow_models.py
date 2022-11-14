import numpy as np
from scipy import stats, optimize, sparse
import copy
import cvxpy as cp
from matplotlib import cm
import networkx as nx
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from multiprocessing import Queue
from matplotlib.patches import ArrowStyle
from matplotlib.colors import LinearSegmentedColormap

class NetworkFlow:
    def __init__(self):
        self.data = None
        self.node_pos = None
        self.graph = Graph()
        self.wdf_model = None
    
    def load_data(self, data, node_pos):
        self.data = data
        self.node_pos = node_pos
        
    def grid_graph(self):
        """
        creates a grid graph
        """
        edge_list = []
        
        for v1 in self.node_pos:
            pos_v1 = self.node_pos[v1]
            
            left_neighbor = [None, np.inf]
            bottom_neighbor = [None, np.inf]

            for v2 in self.node_pos:
                pos_v2 = self.node_pos[v2]
                dist = pos_v1 - pos_v2
                if dist[1] == 0 and dist[0] > 0 and dist[0] < left_neighbor[1]:
                    left_neighbor[0] = v2
                    left_neighbor[1] = dist[0]

                elif dist[0] == 0 and dist[1] > 0 and dist[1] < bottom_neighbor[1]:
                    bottom_neighbor[0] = v2
                    bottom_neighbor[1] = dist[1]

            if left_neighbor[0] is not None:
                edge_list.append((np.min([v1, left_neighbor[0]]), np.max([v1, left_neighbor[0]]), {'weight': 1}))
            if bottom_neighbor[0] is not None:
                edge_list.append((np.min([v1, bottom_neighbor[0]]), np.max([v1, bottom_neighbor[0]]), {'weight': 1}))
                
        self.graph.generate_from_edge_list(edge_list)

    def nn_graph(self, n_neighbors=8):
        """
        Creates a nearest neighbor graph
        """
        edge_list = []

        for v1 in self.node_pos:
            pos_v1 = self.node_pos[v1]

            dist_to_v1 = []
            v2_list = []
            for v2 in self.node_pos:
                pos_v2 = self.node_pos[v2]
                dist = np.sum((pos_v1 - pos_v2)**2)
                dist_to_v1.append(dist)
                v2_list.append(v2)
            idx_sort = np.argsort(dist_to_v1)
            v2_list = np.array(v2_list)
            v1_neighbors = v2_list[idx_sort[1:n_neighbors+1]]

            for v2 in v1_neighbors:
                if (np.min([v1, v2]), np.max([v1, v2]), {'weight': 1}) not in edge_list:
                    edge_list.append((np.min([v1, v2]), np.max([v1, v2]), {'weight': 1}))

        self.graph.generate_from_edge_list(edge_list)

    def proximity_graph(self, dist_th=5):
        """
        Creates graph based on node proximity. All nodes that are less or equal
        than dist_th away from each other are connected
        """
        edge_list = []
        for v1 in self.node_pos:
            pos_v1 = self.node_pos[v1]

            v2_list = []
            for v2 in self.node_pos:
                pos_v2 = self.node_pos[v2]
                dist = np.sqrt(np.sum((pos_v1 - pos_v2)**2))
                if dist <= dist_th and dist != 0:
                    v2_list.append(v2)

            for v2 in v2_list:
                if (np.min([v1, v2]), np.max([v1, v2]), {'weight': 1}) not in edge_list:
                    edge_list.append((np.min([v1, v2]), np.max([v1, v2]), {'weight': 1}))

        self.graph.generate_from_edge_list(edge_list)

    def custom_proximity_graph(self, dist_dct=None):
        """
        dist_dct = {
            dist1 : [d1_n1, d1_n2, ...],
            dist2 : [d2_n1, d2_n2, ...],
            ...
        }
        """

        if dist_dct is None:
            dist_dct = {
                1.5 : [],
                2.9 : []
            }
            for v1 in self.node_pos:
                v1_neighbor_dist = []
                for v2 in self.node_pos:
                    dist = np.sqrt(np.sum((self.node_pos[v1] - self.node_pos[v2])**2))
                    if dist > 0:
                        v1_neighbor_dist.append(dist)
                if np.min(v1_neighbor_dist) < 1.5:
                    dist_dct[1.5].append(v1)
                else:
                    dist_dct[2.9].append(v1)

        edge_list = []
        for dist_th in dist_dct:
            for v1 in dist_dct[dist_th]:
                pos_v1 = self.node_pos[v1]

                v2_list = []
                for v2 in self.node_pos:
                    pos_v2 = self.node_pos[v2]
                    dist = np.sqrt(np.sum((pos_v1 - pos_v2)**2))
                    if dist <= dist_th and dist != 0:
                        v2_list.append(v2)

                for v2 in v2_list:
                    if (np.min([v1, v2]), np.max([v1, v2]), {'weight': 1}) not in edge_list:
                        edge_list.append((np.min([v1, v2]), np.max([v1, v2]), {'weight': 1}))

        self.graph.generate_from_edge_list(edge_list)

    def init_wdf(self, variance=0, verbose=False):
        if self.graph.B1 is None:
            print('Specify graph first!')
        else:
            self.wdf_model = WDFModel(self.graph.B1)
            self.wdf_model.set_model_data(self.data, variance=variance, verbose=verbose)

    def fit_wdf(self, constraints, clip_weights=0, use_cvxpy=True):
        self.wdf_model.fit(constraints=constraints, clip_weights=clip_weights,
                           use_cvxpy=use_cvxpy)


class WDFModel:
    """
    Weighted Diffusion Flow Model
    """
    def __init__(self, B1):
        self.B1 = B1
        self.N = np.shape(self.B1)[0]
        self.E = np.shape(self.B1)[1]
        self.P = np.zeros((self.N + self.E, self.N + self.E))
        self.P_sparse_norm = sparse.csc_matrix(self.P)
        self.b = np.zeros(self.N + self.E)
        self.b_norm = np.zeros(self.N + self.E)
        self.norm_scale = 1
        self.signal_energy = 0
        self.optimization_problem = None
        self.weights = None

    def reset_model_data(self):
        self.P = np.zeros((self.N + self.E, self.N + self.E))
        self.P_sparse_norm = sparse.csc_matrix(self.P)
        self.b = np.zeros(self.N + self.E)
        self.b_norm = np.zeros(self.N + self.E)
        self.norm_scale = 1
        self.signal_energy = 0

    def set_model_data(self, s0_mtx, variance=0, verbose=False):
        self.reset_model_data()
        if len(s0_mtx.shape) == 2:
            S = s0_mtx[:,:-1]
            Sp = s0_mtx[:,1:]
        elif len(s0_mtx.shape) == 3:
            S = np.concatenate(s0_mtx[:,:,:-1], axis=1)
            Sp = np.concatenate(s0_mtx[:,:,1:], axis=1)
        else:
            print(f'Data matrix has unsupported shape {s0_mtx.shape}')

        vi = np.zeros(self.N + self.E)
        ei = np.zeros(self.N + self.E)

        for i in tqdm(range(len(S.T)), disable=not verbose):
            si = S[:,i]
            si_grad = np.dot(self.B1.T, si)
            sip = Sp[:,i]

            Pi1 = np.concatenate((np.diag(si), np.zeros((self.N, self.E))), axis=1)
            Pi2 = np.dot(self.B1, np.concatenate((np.zeros((self.E, self.N)), np.diag(si_grad)), axis=1))

            Pi = np.dot((Pi1 + Pi2).T, Pi1 + Pi2)
            self.P += Pi

            vi[:self.N] = sip * si
            ei[self.N:] = np.dot(sip, self.B1) * si_grad

            self.b += vi + ei
            self.signal_energy += np.sum(sip**2)

        self.P /= len(S.T)
        self.b /= len(S.T)

        # correcting data matrix P for the noise variance
        L1 = self.B1.T @ self.B1
        P_var = variance * np.block([
            [np.identity(self.N), abs(self.B1)],
            [abs(self.B1.T), abs(L1) + np.diag(np.diag(L1))]
        ])

        self.P -= P_var

        # normalize the data
        self.norm_scale = np.median(np.concatenate((abs(self.P[np.nonzero(self.P)]).flatten(),
                                                    abs(self.b))))
        self.P_sparse_norm = sparse.csc_matrix(self.P / self.norm_scale)
        self.b_norm = self.b / self.norm_scale

    def fit(self, constraints, clip_weights=0, use_cvxpy=True):
        if use_cvxpy:
            P_sparse_cp = cp.atoms.affine.wraps.psd_wrap(self.P_sparse_norm)
            w = cp.Variable(self.N + self.E)
            cost = 0.5 * cp.quad_form(w, P_sparse_cp) - self.b_norm.T @ w
            num_constraints = len(constraints)
            if num_constraints == 0:
                prob = cp.Problem(cp.Minimize(cost))
            else:
                cp_constraints = []
                for c in constraints:
                    cp_constraints.append(c(w))
                prob = cp.Problem(cp.Minimize(cost), cp_constraints)
            prob.solve(solver='OSQP', eps_abs=1e-5, eps_rel=1e-5)

            self.optimization_problem = prob
            if clip_weights is None:
                self.weights = w.value
            else:
                self.weights = w.value.clip(min=clip_weights)
        else:
            # note that this will always solve the unconstrained problem
            self.weights = np.linalg.pinv(self.P) @ self.b

    def get_abs_error(self):
        abs_err = self.signal_energy + self.weights.T @ self.P @ self.weights - 2 * self.b.T @ self.weights
        return abs_err

    def get_rel_improvement(self):
        abs_err = self.get_abs_error()
        rel_err = (self.signal_energy - abs_err) / self.signal_energy
        return rel_err
    
class Graph:
    """
    Generic graph class that supports edge signals
    """
    def __init__(self):
        self.B1 = None
        self.A = None
        self.edge_list = None
        self.s1 = None
        self.s0 = None
        self.nodes = None
        self.neighbors = None
        self.L1_low = None
        self.lmda_L1_low = None
        self.V_L1_low = None
        self.node_pos = None
        self.node_names = None
        
    def generate_from_edge_list(self, edge_list):
        """
        edge_list should have the structure [(from_node, to_node, {'weight': w})].
        Node labels should start at 0.
        The reference orientation for B1 is from the node with lower index to the node with higher index
        """
        # get number of nodes
        nodes = []
        for n in edge_list:
            nodes.append(n[0])
            nodes.append(n[1])
        N = np.max(nodes) + 1
        self.nodes = np.unique(nodes)
        
        # edges:
        E = len(edge_list)
        
        # node to incidence matrix B1, edge signal s1, and edge list
        self.B1 = np.zeros((N, E))
        self.s1 = np.zeros(E)
        self.edge_list = []
        for i, edge in enumerate(edge_list):
            if edge[0] < edge[1]:
                self.B1[edge[0],i] = -1
                self.B1[edge[1],i] = 1
                self.s1[i] = edge[2]['weight']
                self.edge_list.append(edge)
            else:
                self.B1[edge[1],i] = -1
                self.B1[edge[0],i] = 1
                self.s1[i] = -edge[2]['weight']
                self.edge_list.append((edge[1], edge[0], {'weight': -edge[2]['weight']}))
            
        # Adjacency matrix A
        self.A = np.dot(self.B1, self.B1.T)

        # neighbor dictionary
        neighbors = {}
        for node in self.nodes:
            n_list = []
            for edge in self.edge_list:
                if node in edge:
                    if node == edge[0]:
                        n_list.append(edge[1])
                    else:
                        n_list.append(edge[0])
            neighbors[node] = n_list
        self.neighbors = neighbors
        
    def set_edge_signal(self, s1):
        if len(s1) != len(self.edge_list):
            raise TypeError('edge signal s1 must have same length as edge_list')
        for i in range(len(s1)):
            self.edge_list[i][2]['weight'] = s1[i]
        self.s1 = s1
        
    def edge_list_correct_direction(self, edge_list=None):
        """
        flips direction of arrow is edge_weight is negative. Useful for plotting edge signals
        """
        if edge_list is None:
            edge_list = self.edge_list
        
        new_edge_list = copy.deepcopy(edge_list)
        for i in range(len(edge_list)):
            if edge_list[i][2]['weight'] < 0:
                from_edge = edge_list[i][1]
                to_edge = edge_list[i][0]
                new_edge_list[i] = (from_edge, to_edge, {'weight': -new_edge_list[i][2]['weight']})

        return new_edge_list

    def get_L1_low(self):
        if self.B1 is not None:
            self.L1_low = np.matmul(self.B1.T, self.B1)
            return self.L1_low
        else:
            print('Generate graph first')
            return None

    def L1_low_decomposition(self):
        if self.L1_low is None:
            self.get_L1_low()

        w, V = np.linalg.eigh(self.L1_low)
        
        # since L1 is symmetric and PSD, w, V should be real
        idx_sort  = np.argsort(abs(w))
        w_sorted = abs(w[idx_sort])
        V_sorted = np.real(V[:,idx_sort])
        
        self.lmda_L1_low = w_sorted
        self.V_L1_low = V_sorted

        return self.lmda_L1_low, self.V_L1_low

    def flow_ft(self, s1):
        if self.lmda_L1_low is None or self.V_L1_low is None:
            self.L1_low_decomposition()

        N_harm = self.B1.shape[1] - self.B1.shape[0] + 1
        V_grad = self.V_L1_low[:, N_harm:]
        V_harm = self.V_L1_low[:, :N_harm]

        S1_grad = V_grad.T @ s1
        S1_harm = V_harm.T @ s1

        lmda_grad = self.lmda_L1_low[N_harm:]

        return lmda_grad, V_grad, S1_grad, V_harm, S1_harm


    def set_node_pos_name(self, node_pos, node_names=None):
        self.node_pos = node_pos
        self.node_names = node_names

class NetworkAnimation:
    def __init__(self, S1_mtx, graph, electrode_pos, electrode_idx, stim1_idx, stim2_idx, m1_nodes, s1_nodes,
                 vmin=None, vmax=None, cmap='default', node_color='lightgray', figsize=(14,8),
                 annotation=True):
        self.graph = graph
        self.S1_mtx = S1_mtx
        self.electrode_pos = electrode_pos
        self.electrode_idx = electrode_idx
        self.stim1_idx = stim1_idx
        self.stim2_idx = stim2_idx
        self.s1_nodes = s1_nodes
        self.m1_nodes = m1_nodes
        self.annotation = annotation

        if cmap == 'default':
            cmap = cm.seismic
            colors = cmap(np.linspace(0.5, 1, cmap.N // 2))
            self.cmap = LinearSegmentedColormap.from_list('Upper Half', colors)
        else:
            self.cmap = cmap

        self.graph.set_edge_signal(self.S1_mtx[:,0])
        self.node_color = node_color
        direction_corr_edge_list = self.graph.edge_list_correct_direction()
        G1 = nx.DiGraph(direction_corr_edge_list)

        if vmin is None:
            self.vmin = 0
        else:
            self.vmin = vmin
        if vmax is None:
            self.vmax = np.quantile(abs(self.S1_mtx), 0.95)
        else:
            self.vmax = vmax

        edge_width = np.ones(len(direction_corr_edge_list)) * 3.0
        for k, e in enumerate(G1.edges):
            if (e[0] in self.m1_nodes and e[1] in self.s1_nodes) or \
            (e[0] in self.s1_nodes and e[1] in self.m1_nodes):
                edge_width[k] = 8.0

        self.arrow_style = ArrowStyle.CurveFilledB(head_length=0.8, head_width=0.3)

        self.fig, self.ax = plt.subplots(figsize=figsize)
        nx.draw(G1, pos=self.electrode_pos, edge_color = nx.to_pandas_edgelist(G1)['weight'],
                width=edge_width, edge_cmap=self.cmap, node_size=300, node_color=self.node_color,
                edge_vmin=self.vmin, edge_vmax=self.vmax, arrowstyle=self.arrow_style)

        self.stim_nodes = []
        if self.stim1_idx is not None:
            self.stim_nodes.append(self.stim1_idx)
        if self.stim2_idx is not None:
            self.stim_nodes.append(self.stim2_idx)

        if len(self.stim_nodes) > 0:
            nodes = nx.draw_networkx_nodes(G1, pos=self.electrode_pos, nodelist=self.stim_nodes,
                                           node_size=700, node_color='black', node_shape='X', linewidths=3)

        if self.electrode_idx is not None:
            if len(self.electrode_idx) == len(self.electrode_pos):
                labels = nx.draw_networkx_labels(G1, pos=self.electrode_pos, labels=self.electrode_idx)

        if self.annotation:
            sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(vmin=self.vmin, vmax=self.vmax))
            sm._A = []
            cbar = plt.colorbar(sm, pad=-0.05, label='normalized flow magnitude', extend='max')
            plt.title(f'Stim + 0 ms')

    def anim_init(self):
        pass

    def animate(self, frame):
        self.ax.clear()
        self.graph.set_edge_signal(self.S1_mtx[:,frame])
        direction_corr_edge_list = self.graph.edge_list_correct_direction()
        G1 = nx.DiGraph(direction_corr_edge_list)

        edge_width = np.ones(len(direction_corr_edge_list)) * 3.0
        for k, e in enumerate(G1.edges):
            if (e[0] in self.m1_nodes and e[1] in self.s1_nodes) or \
                    (e[0] in self.s1_nodes and e[1] in self.m1_nodes):
                edge_width[k] = 8.0

        nx.draw(G1, pos=self.electrode_pos, edge_color = nx.to_pandas_edgelist(G1)['weight'],
                width=edge_width, edge_cmap=self.cmap, node_size=300, node_color=self.node_color,
                edge_vmin=self.vmin, edge_vmax=self.vmax, arrowstyle=self.arrow_style)

        if len(self.stim_nodes) > 0:
            nodes = nx.draw_networkx_nodes(G1, pos=self.electrode_pos, nodelist=self.stim_nodes,
                                           node_size=700, node_color='black', node_shape='X', linewidths=3)

        if self.electrode_idx is not None:
            if len(self.electrode_idx) == len(self.electrode_pos):
                labels = nx.draw_networkx_labels(G1, pos=self.electrode_pos, labels=self.electrode_idx)

        if self.annotation:
            plt.title(f'Stim + {frame} ms')


class TSCSignal(Graph):
    """
    Class for time dependent signals on simplicial complexes (SC)
    For now the focus is on signals defined on 2-complex. i.e. edge flow
    signals
    """
    def __init__(self, edge_list):
        super.__init__()
        self.generate_from_edge_list(edge_list)
        self.S = None # SC signal

    def set_flow_signal(self, flow_signal):
        """
        :param flow_signal: E x T numpy array with each column being one flow signal.
            T = number of time steps
        """
        self.S = flow_signal

    def get_jft(self, flow_laplacian='l1_low'):
        t = self.S.shape[1]
        E = self.S.shape[0]
        T = len(t)

        if flow_laplacian == 'l1_low':
            lmda = np.array([l if l >1e-10 else 0.0 for l in self.lmda_L1_low])
            self.w = 2 * np.pi * t / T
            self.Ut = np.zeros((T, T), dtype=np.complex128)
            for i, tk in enumerate(t):
                for j, wk in enumerate(self.w):
                    self.Ut[i,j] = np.exp(-1j*wk*tk) / np.sqrt(T)
            self.Ut = np.matrix(self.Ut).H
            self.Uj = np.kron(self.Ut, self.V_L1_low)

            s_flat = self.S.flatten(order='F').reshape(-1,1)
            s_jft = (self.Uj @ s_flat).reshape(T, E).T
            return s_jft
        else:
            # TODO: implement different flow laplacians
            pass

    def fir_filter(self, w_c, lmda_c):
        pass
