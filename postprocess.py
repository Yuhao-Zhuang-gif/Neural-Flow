import numpy as np
import pickle
from preprocess import ECoGPreprocess
from network_flow_models import NetworkAnimation
from matplotlib import cm

class FlowLoader:
    def __init__(self, base_path_model, base_path_ecog, session, block_type, block_id, bp_filter=[]):
        """
        :param base_path_model: str
            folder containing the fitted model parameter. The model parameter are stored in a pickle file <session>.pkl
        :param base_path_ecog: str
            folder containing the ECOG data. Each session has its separate folder names <session>
        :param session: str
            name of the session
        :param block_type: str
            'cond', 'rec', or 'test' for conditioning, recording or testing block respectively
        :param block_id: int
            block number. There are 6 blocks each for recording and testing and 5 blocks for conditioning.
        """

        # loading the model parameter for the given session
        self.block_type = block_type
        self.block_id = block_id

        file_name = base_path_model + session + '.pkl'
        with open(file_name, "rb") as input_file:
            self.model_param = pickle.load(input_file)

        self.graph = self.model_param['graph'] # the graph
        self.B1 = self.graph.B1 # node to incidence matrix
        self.E = self.B1.shape[1] # number of edges
        self.N = self.B1.shape[0] # number of nodes

        # number of frames of flow matrix
        self.T = self.model_param['model_fit_win_cond']

        # load the ECoG data
        self.preprocess = ECoGPreprocess(session, base_path_ecog)
        if block_type == 'cond':
            data = self.preprocess.load_measurement_block("CondBlock" + str(block_id))
        elif block_type == 'rec':
            print('Currently only cond blocks are supported')
            pass
        self.data_clean = self.preprocess.remove_bad_channels()
        if len(bp_filter) == 2:
            self.data_clean = self.preprocess.bp_filter(fmin=bp_filter[0], fmax=bp_filter[1])

        # get number of laser stim
        self.stim1_flag = self.model_param['stim1_flag']
        self.stim2_flag = self.model_param['stim2_flag']

        if self.stim1_flag:
            self.num_stim = len(self.preprocess.stim1)
            self.which_stim = 1
        elif self.stim2_flag:
            self.num_stim = len(self.preprocess.stim2)
            self.which_stim = 2

        # load the nodes in M1 and S1, respectively
        self.electrode_names = self.model_param['electrode_idx']
        m1_nodes = self.model_param['table_of_experiment']['m1_sites']
        m1_nodes = [int(n) for n in m1_nodes.split(',')]

        self.s1_nodes_good = []
        self.m1_nodes_good = []
        for n in self.electrode_names:
            if self.electrode_names[n] in m1_nodes:
                self.m1_nodes_good.append(n)
            else:
                self.s1_nodes_good.append(n)

        # load location of stimulation
        self.stim1_idx = self.model_param['stim1_idx']
        self.stim2_idx = self.model_param['stim2_idx']

    def get_num_model_fits(self):
        """
        :return: int
            how many times the model has been fitted to within the given block
        """
        return self.num_stim

    def get_single_flow_mtx(self, index):
        """
        :param index: int
             indicates the model fitting instance
        :return: np.ndarray
            2-D numpy array with dimension self.E x self.T. Each column is a single flow snapshot. Each row on the other hand is the flow time series at a single edge.
        """
        data_short, t, stim_time, idx = self.preprocess.select_data_around_stim(index, self.T, which_stim=self.which_stim)
        w1 = self.model_param[self.block_type][self.block_id]['unconstrained']['model_param_train'][self.N:]

        flow = np.dot(np.diag(-w1), np.dot(self.B1.T, data_short))
        return flow

    def generate_animation(self, flow_matrix, node_names='default', stim1_idx='default',
                           stim2_idx='default', m1_nodes='default', s1_nodes='default',
                           cmap='default', node_color='default', vmin=None, vmax=None,
                           figsize=(14,8), annotation=True):
        """
        :param flow_matrix: np.ndarray
            2-D numpy array with dimension self.E x self.T. Each column is a single flow snapshot. Each row on the other hand is the flow time series at a single edge.
        :return: NetworkAnimation
            an instance of NetworkAnimation
        """
        electrode_pos = self.model_param['electrode_pos']

        if node_names == 'default':
            node_names = self.electrode_names
        if stim1_idx == 'default':
            stim1_idx = self.stim1_idx
        if stim2_idx == 'default':
            stim2_idx = self.stim2_idx
        if s1_nodes == 'default':
            s1_nodes = self.s1_nodes_good
        if m1_nodes == 'default':
            m1_nodes = self.m1_nodes_good
        if node_color == 'default':
            node_color='lightgray'


        nf_anim = NetworkAnimation(
            flow_matrix, graph=self.graph, electrode_pos=electrode_pos,
            electrode_idx=node_names, stim1_idx=stim1_idx, stim2_idx=stim2_idx,
            m1_nodes=m1_nodes, s1_nodes=s1_nodes, cmap=cmap, node_color=node_color,
            vmin=vmin, vmax=vmax, figsize=figsize, annotation=annotation
        )

        return nf_anim

    def get_edge_affiliation(self):
        """
        :return: (list, list, list)
            returns indices of (S1, M1, across) edges
        """
        edge_list = self.graph.edge_list

        idx_S1 = []
        idx_M1 = []
        idx_across = []

        for k, e in enumerate(edge_list):
            if e[0] in self.s1_nodes_good and e[1] in self.s1_nodes_good:
                idx_S1.append(k)
            elif e[0] in self.m1_nodes_good and e[1] in self.m1_nodes_good:
                idx_M1.append(k)
            else:
                idx_across.append(k)

        return idx_S1, idx_M1, idx_across