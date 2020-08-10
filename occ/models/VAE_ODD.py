from occ.models.abstract_occ_model import abstract_occ_model
from occ.models.AutoEncoderODD import AutoEncoderODD
from pyod.models.vae import VAE 

class VAE_ODD(AutoEncoderODD):
    def __init__(self, hidden_neurons, nu, epochs, batch_size=32):
        if len(hidden_neurons) % 2 == 0:
            print("The number of layers must be an odd number(2n+1).")
            sys.exit()
        encoder = hidden_neurons[0:len(hidden_neurons)//2]
        latent = hidden_neurons[len(hidden_neurons)//2]
        decoder = hidden_neurons[len(hidden_neurons)//2+1:len(hidden_neurons)]
        self.model = VAE(encoder_neurons=encoder,
                         decoder_neurons=decoder,
                         latent_dim=latent,
                         contamination=nu,
                         epochs=epochs,
                         batch_size=batch_size
                        )