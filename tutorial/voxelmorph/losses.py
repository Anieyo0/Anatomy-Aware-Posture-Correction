import torch
import torch.nn.functional as F
import numpy as np
import math


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()




# ---

class AnatomyWeightedMSE:
    def __init__(self):
        pass

    def loss(self, y_true, y_pred, anatomical_map):
        """
        Parameters:
            y_true: Ground truth image (B, C, H, W)
            y_pred: Predicted image (B, C, H, W)
            Anatomical_map: Anatomical weight map (B, 1, H, W) ∈ [0, 1]
        Returns:
            Weighted MSE loss (scalar)
        """
        pixel_diff = (y_true - y_pred) ** 2
        weighted_diff = pixel_diff * anatomical_map
        return torch.mean(weighted_diff)


class RegistrationLoss:
    def __init__(
        self,
        total_iterations=None,
        use_mse=True,
        use_anatomical_mse=False,
        use_plain_edge=False,
        use_anatomical_edge=False,
        use_grad=True,
        adaptive_mse_transition=False,
        adaptive_edge_transition=False,
        edge_weight=1,
        grad_weight=0.5,
    ):
        self.use_mse = use_mse
        self.use_anatomical_mse = use_anatomical_mse
        self.use_plain_edge = use_plain_edge
        self.use_anatomical_edge = use_anatomical_edge
        self.use_grad = use_grad

        self.adaptive_mse_transition = adaptive_mse_transition
        self.adaptive_edge_transition = adaptive_edge_transition
        self.edge_weight = edge_weight
        self.total_iterations = total_iterations
        self.current_iteration = 0

        self.mse = MSE()
        self.anatomy_mse = AnatomyWeightedMSE()
        self.grad_loss = Grad(penalty='l2', loss_mult=grad_weight)

        if (self.adaptive_mse_transition or self.adaptive_edge_transition) and self.total_iterations is None:
            raise ValueError("total_iterations must be set when adaptive transitions are enabled.")

    def smooth_weights(self, iteration):
        """
        Returns:
            w_mse: float (always 1.0)
            w_anatomy: float, increases from 0 to 1 as training progresses
        """
        if self.total_iterations is None:
            raise ValueError("total_iterations must be set when adaptive transitions are enabled.")

        # Schedule parameters
        t = iteration / self.total_iterations  # normalized progress [0, 1]
        transition_point = 0.3  # start increasing anatomical weight after 30% progress
        sharpness = 10          # steeper = sharper transition curve

        w_anatomy = 0.5 * (np.tanh((t - transition_point) * sharpness) + 1)
        w_mse = 1.0  # fixed throughout

        return w_mse, w_anatomy

    def plain_edge_mse(self, y_true_edge, y_pred_edge):
        return torch.mean((y_true_edge - y_pred_edge) ** 2)

    def anatomy_edge_mse(self, y_true_edge, y_pred_edge, anatomical_map):
        diff = (y_true_edge - y_pred_edge) ** 2
        return torch.mean(diff * anatomical_map)

    def loss(self, y_true, y_pred, anatomical_map, flow, y_true_edge=None, y_pred_edge=None):
        total = 0.0

        # MSE weights
        if self.adaptive_mse_transition:
            w_mse, w_anatomy = self.smooth_weights(self.current_iteration)
        else:
            w_mse = 1.0 if self.use_mse else 0.0
            w_anatomy = 1.0 if self.use_anatomical_mse else 0.0

        # Edge weights
        if self.adaptive_edge_transition:
            w_edge_plain, w_edge_anatomy = self.smooth_weights(self.current_iteration)
        else:
            w_edge_plain = 1.0 if self.use_plain_edge else 0.0
            w_edge_anatomy = 1.0 if self.use_anatomical_edge else 0.0

        # --- MSE
        raw_mse = self.mse.loss(y_true, y_pred) if w_mse > 0 else 0.0
        loss_mse_component = w_mse * raw_mse
        total += loss_mse_component

        # --- Anatomical MSE
        raw_anatomy = self.anatomy_mse.loss(y_true, y_pred, anatomical_map) if w_anatomy > 0 else 0.0
        loss_anatomy_component = w_anatomy * raw_anatomy
        total += loss_anatomy_component

        # --- Edge Losses
        if w_edge_plain > 0 or w_edge_anatomy > 0:
            if y_true_edge is None or y_pred_edge is None:
                raise ValueError("Edge inputs must be provided if any edge loss is enabled.")

            raw_edge_plain = self.plain_edge_mse(y_true_edge, y_pred_edge) if w_edge_plain > 0 else 0.0
            raw_edge_anatomy = self.anatomy_edge_mse(y_true_edge, y_pred_edge, anatomical_map) if w_edge_anatomy > 0 else 0.0

            loss_edge_plain_component = self.edge_weight * w_edge_plain * raw_edge_plain
            loss_edge_anatomy_component = self.edge_weight * w_edge_anatomy * raw_edge_anatomy
        else:
            raw_edge_plain = raw_edge_anatomy = 0.0
            loss_edge_plain_component = loss_edge_anatomy_component = 0.0

        total += loss_edge_plain_component + loss_edge_anatomy_component

        # --- Grad Loss
        raw_grad = self.grad_loss.loss(None, flow) if self.use_grad else 0.0
        loss_grad_component = self.grad_loss.loss_mult * raw_grad
        total += loss_grad_component

        return (
            total,
            loss_mse_component, raw_mse, w_mse,
            loss_anatomy_component, raw_anatomy, w_anatomy,
            loss_edge_plain_component, raw_edge_plain, w_edge_plain,
            loss_edge_anatomy_component, raw_edge_anatomy, w_edge_anatomy,
            loss_grad_component, raw_grad, self.grad_loss.loss_mult if self.use_grad else 0.0
        )
