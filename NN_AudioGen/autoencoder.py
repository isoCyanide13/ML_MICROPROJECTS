class Autoencoder:
    """
    Autoencoder represents a Deep convolutional autoencoder architecture woth
    mirrored encoder and decoder components
    """
    
def __init__(self,
            input_shape,
            conv_filters,
            conv_kernals,
            conv_strides,
            latent_space_dim):
    self.input_shape = input_shape
    self.conv_filters = conv_filters
    self.conv_kernals = conv_kernals
    self.conv_strides = conv_strides
    self.latent_space_dim = latent_space_dim
    
    self.encoder = None
    self.decoder = None
    self.model = None
    
    
    pass