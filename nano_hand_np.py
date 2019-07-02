import smpl_np
import numpy as np
import pickle


class NANOHandModel(smpl_np.SMPLModel):
    def __init__(self, model_path, ncomps=6, flat_hand_mean=False):
        """
        SMPL model.

        Parameter:
        ---------
        model_path: Path to the SMPL model parameters, pre-processed by
        `preprocess.py`.

        """
        super(NANOHandModel, self).__init__(model_path)
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
            self.hands_components = params['hands_components']
            self.hands_mean = params['hands_mean']

        self.pose_shape = [15, 3]

        self.pose = np.zeros(self.pose_shape)
        self.pose_comps = np.zeros(ncomps)
        self.flat_hand_mean = flat_hand_mean

        self.selected_components = self.hands_components[:ncomps]

    def set_params(self, pose_comps=None, beta=None, trans=None, rot=None):
        """
        Set pose, shape, and/or translation parameters of SMPL model. Verices of the
        model will be updated and returned.

        Prameters:
        ---------
        pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.

        beta: Parameter for model shape. A vector of shape [10]. Coefficients for
        PCA component. Only 10 components were released by MPI.

        trans: Global translation of shape [3].

        Return:
        ------
        Updated vertices.

        """
        if pose_comps is not None:
            self.pose_comps = pose_comps
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans

        if rot is not None:
            self.pose = np.concatenate((rot, (self.hands_mean + self.pose_comps.dot(self.selected_components))))

        self.update()
        return self.verts


if __name__ == '__main__':
    nano = NANOHandModel('./nano_hand_model.pkl')
    np.random.seed(9608)
    pose_comps = np.array([-0.42671473, -0.85829819, -0.50662164, +1.97374622, -0.84298473, -1.29958491])
    beta = (np.random.rand(*nano.beta_shape)) * .03
    trans = np.zeros(nano.trans_shape)
    rot = np.array([.0, .0, .0])
    nano.set_params(beta=beta, pose_comps=pose_comps, trans=trans, rot=rot)
    nano.save_to_obj('./nano_hand_np.obj')
