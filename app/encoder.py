from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def encoder(input_dim, latent_dim=20):

    encoder_input = Input(shape=(input_dim,), name="encoder_input")
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(encoder_input)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    latent = Dense(latent_dim,activation='linear',name='latent_space')(x)
    encoder_model = Model(encoder_input, latent, name="encoder")



    decoder_input = Input(shape=(latent_dim,), name="decoder_input")
    x = Dense(128,activation='relu')(decoder_input)
    x = Dense(512,activation='relu')(x)
    decoder_output = Dense(input_dim, activation="linear")(x)
    decoder_model = Model(decoder_input, decoder_output, name="decoder")



    autoencoder_input = Input(shape=(input_dim,))
    encoded = encoder_model(autoencoder_input)
    decoded = decoder_model(encoded)
    autoencoder_model = Model(autoencoder_input, decoded)
    autoencoder_model.compile(optimizer="adam", loss="huber", jit_compile=False)

    # print(latent_targets.shape)

    return autoencoder_model, encoder_model, decoder_model