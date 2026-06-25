from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2

import numpy as np

def predictor(input_dim,  latent_dim):
    
    predictor_input = Input(shape=(input_dim,))
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(predictor_input)
    # x = Dropout(0.4)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    # x = Dropout(0.3)(x)
    predictor_output = Dense(latent_dim, activation='linear')(x)
    predictor_model = Model(inputs=predictor_input, outputs=predictor_output)
    predictor_model.compile(optimizer='adam', loss='mse')
    # predictor.fit(X_scaled, latent_targets, epochs=500, batch_size=8, verbose=1, callbacks=[early_stop])

    return predictor_model