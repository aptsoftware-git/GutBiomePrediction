"""In-memory TensorFlow training without rebuilding input pipelines each epoch."""
import numpy as np
from runtime import tf


def fit_in_memory(model, inputs, targets, *, epochs=500, batch_size=32,
                  patience=25, min_delta=1e-5, progress=None):
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    targets = tf.convert_to_tensor(targets, dtype=tf.float32)
    count = int(inputs.shape[0])
    batch_size = min(batch_size, count)
    loss_function = tf.keras.losses.get(model.loss)
    model.optimizer.build(model.trainable_variables)

    # Compile once per model; avoid XLA compilation for every validation fold.
    @tf.function(reduce_retracing=True, jit_compile=False)
    def train_epoch():
        order = tf.random.shuffle(tf.range(count))
        total = tf.constant(0.0)
        for start in tf.range(0, count, batch_size):
            indexes = order[start:start + batch_size]
            x = tf.gather(inputs, indexes)
            y = tf.gather(targets, indexes)
            with tf.GradientTape() as tape:
                predicted = model(x, training=True)
                loss = tf.reduce_mean(loss_function(y, predicted))
                if model.losses:
                    loss += tf.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total += loss * tf.cast(tf.size(indexes), tf.float32)
        return total / tf.cast(count, tf.float32)

    best_loss = float("inf")
    best_weights = None
    wait = 0
    for epoch in range(1, epochs + 1):
        loss = float(train_epoch().numpy())
        if not np.isfinite(loss):
            raise ValueError("Training produced a non-finite loss; check the input data.")
        if loss < best_loss - min_delta:
            best_loss = loss
            best_weights = model.get_weights()
            wait = 0
        else:
            wait += 1
        if progress and (epoch == 1 or epoch % 25 == 0):
            progress(f"Epoch {epoch}/{epochs}, loss {loss:.5f}")
        if wait >= patience:
            break
    model.set_weights(best_weights)
    return {"epochs": epoch, "best_loss": best_loss}
