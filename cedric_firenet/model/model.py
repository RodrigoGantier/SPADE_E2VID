from cedric_firenet.base import BaseModel
import torch.nn as nn
import torch
from cedric_firenet.model.unet import UNet, UNetRecurrent, UNetFire
from os.path import join
from cedric_firenet.model.submodules import ConvLSTM, ResidualBlock, ConvLayer, UpsampleConvLayer, TransposedConvLayer


class BaseE2VID(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        assert('num_bins' in config)
        self.num_bins = int(config['num_bins'])  # number of bins in the voxel grid event tensor

        try:
            self.skip_type = str(config['skip_type'])
        except KeyError:
            self.skip_type = 'sum'

        try:
            self.num_encoders = int(config['num_encoders'])
        except KeyError:
            self.num_encoders = 4

        try:
            self.base_num_channels = int(config['base_num_channels'])
        except KeyError:
            self.base_num_channels = 32

        try:
            self.num_residual_blocks = int(config['num_residual_blocks'])
        except KeyError:
            self.num_residual_blocks = 2

        try:
            self.norm = str(config['norm'])
        except KeyError:
            self.norm = None

        try:
            self.use_upsample_conv = bool(config['use_upsample_conv'])
        except KeyError:
            self.use_upsample_conv = True


class E2VID(BaseE2VID):
    def __init__(self, config):
        super(E2VID, self).__init__(config)

        self.unet = UNet(num_input_channels=self.num_bins,
                         num_output_channels=1,
                         skip_type=self.skip_type,
                         activation='sigmoid',
                         num_encoders=self.num_encoders,
                         base_num_channels=self.base_num_channels,
                         num_residual_blocks=self.num_residual_blocks,
                         norm=self.norm,
                         use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states=None):
        """
        :param event_tensor: N x num_bins x H x W
        :return: a predicted image of size N x 1 x H x W, taking values in [0,1].
        """
        return self.unet.forward(event_tensor), None


class E2VIDRecurrent(BaseE2VID):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, config):
        super(E2VIDRecurrent, self).__init__(config)

        try:
            self.recurrent_block_type = str(config['recurrent_block_type'])
        except KeyError:
            self.recurrent_block_type = 'convlstm'  # or 'convgru'

        self.unetrecurrent = UNetRecurrent(num_input_channels=self.num_bins,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """
        img_pred, states = self.unetrecurrent.forward(event_tensor, prev_states)
        return img_pred, states


class FireNet(BaseE2VID):
    """
    Model from the paper: "Fast Image Reconstruction with an Event Camera", Scheerlinck et. al., 2019.
    The model is essentially a lighter version of E2VID, which runs faster (~2-3x faster) and has considerably less parameters (~200x less).
    However, the reconstructions are not as high quality as E2VID: they suffer from smearing artefacts, and initialization takes longer.
    """
    def __init__(self, config):
        super().__init__(config)
        self.recurrent_block_type = str(config.get('recurrent_block_type', 'convgru'))
        kernel_size = config.get('kernel_size', 3)
        recurrent_blocks = config.get('recurrent_blocks', {'resblock': [0]})
        self.net = UNetFire(self.num_bins,
                            num_output_channels=1,
                            skip_type=self.skip_type,
                            recurrent_block_type=self.recurrent_block_type,
                            base_num_channels=self.base_num_channels,
                            num_residual_blocks=self.num_residual_blocks,
                            norm=self.norm,
                            kernel_size=kernel_size,
                            recurrent_blocks=recurrent_blocks)

    def forward(self, event_tensor, prev_states):
        img, states = self.net.forward(event_tensor, prev_states)
        return img, states
