"""
clf_keras_v2.py — moteur Keras / Keras Tuner de clf_core_v2.run_cv.

Architectures, espace de recherche, Hyperband et entraînement d'un pli : code COPIÉ de loso_pipeline.py /
udep_pipeline.py (mêmes constantes, mêmes builders, mêmes callbacks). Seuls ajouts :
  • tuning refait dans chaque pli (voir clf_core_v2, tune_mode) → aucun pli de test n'influence les hyperparamètres ;
  • hyperparamètres manuels (FixedHP) ;
  • rappels de progression / annulation ;
  • libération de la mémoire entre deux plis.
Ce module importe TensorFlow : il ne tourne que dans le processus de travail (clf_worker_v2.py), jamais dans Flask.
"""
import os
import random as _random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

# ── constantes (identiques aux scripts d'origine) ──
MAX_EPOCHS = 20
HYPERBAND_ITERATIONS = 2
HYPERBAND_FACTOR = 3
EARLY_STOPPING_PATIENCE = 3

DENSE_N_LAYERS = (1, 3)
DENSE_UNITS = [32, 64, 128, 256]
CNN_N_BLOCKS = (1, 3)
CNN_FILTERS = [16, 32, 64, 128]
CNN_KERNELS = [3, 5, 7]
LSTM_N_LAYERS = (1, 3)
LSTM_UNITS = [16, 32, 64, 128]
DROPOUT_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
LEARNING_RATE_MIN = 1e-4
LEARNING_RATE_MAX = 1e-2
BATCH_SIZES = [16, 32, 64]


def seed_everything(run_seed):
    """RUN_SEED : init des poids, tuning, shuffle des batchs (découplée de la graine des plis)."""
    np.random.seed(run_seed)
    tf.random.set_seed(run_seed)
    _random.seed(run_seed)


# ═══════════════════════════════════════════════════════════
#  BUILDERS  (copie conforme)
# ═══════════════════════════════════════════════════════════
def build_dense_model(hp, input_shape, n_classes):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())

    n_layers = hp.Int("n_layers", DENSE_N_LAYERS[0], DENSE_N_LAYERS[1])
    for i in range(n_layers):
        units = hp.Choice(f"units_{i}", DENSE_UNITS)
        dropout = hp.Choice(f"dropout_{i}", DROPOUT_VALUES)
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(n_classes, activation="softmax"))

    lr = hp.Float("lr", LEARNING_RATE_MIN, LEARNING_RATE_MAX, sampling="log")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn_model(hp, input_shape, n_classes):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    n_blocks = hp.Int("n_conv_blocks", CNN_N_BLOCKS[0], CNN_N_BLOCKS[1])
    for i in range(n_blocks):
        filters = hp.Choice(f"filters_{i}", CNN_FILTERS)
        kernel = hp.Choice(f"kernel_{i}", CNN_KERNELS)
        dropout = hp.Choice(f"dropout_{i}", DROPOUT_VALUES)
        model.add(layers.Conv1D(filters, kernel, activation="relu", padding="same"))
        model.add(layers.MaxPooling1D(pool_size=2, padding="same"))
        model.add(layers.Dropout(dropout))

    model.add(layers.GlobalAveragePooling1D())
    dense_units = hp.Choice("dense_units", [32, 64, 96, 128])
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dense(n_classes, activation="softmax"))

    lr = hp.Float("lr", LEARNING_RATE_MIN, LEARNING_RATE_MAX, sampling="log")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm_model(hp, input_shape, n_classes):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    n_layers = hp.Int("n_lstm_layers", LSTM_N_LAYERS[0], LSTM_N_LAYERS[1])
    for i in range(n_layers):
        units = hp.Choice(f"lstm_units_{i}", LSTM_UNITS)
        dropout = hp.Choice(f"dropout_{i}", DROPOUT_VALUES)
        return_sequences = (i < n_layers - 1)
        model.add(layers.Bidirectional(layers.LSTM(units, return_sequences=return_sequences)))
        model.add(layers.Dropout(dropout))

    dense_units = hp.Choice("dense_units", [16, 32, 48, 64])
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dense(n_classes, activation="softmax"))

    lr = hp.Float("lr", LEARNING_RATE_MIN, LEARNING_RATE_MAX, sampling="log")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


ARCH_BUILDERS = {"dense": build_dense_model, "cnn": build_cnn_model, "lstm": build_lstm_model}


class GestureHyperModel(kt.HyperModel):
    def __init__(self, arch, input_shape, n_classes):
        self.arch = arch
        self.input_shape = input_shape
        self.n_classes = n_classes

    def build(self, hp):
        return ARCH_BUILDERS[self.arch](hp, self.input_shape, self.n_classes)

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice("batch_size", BATCH_SIZES)
        return model.fit(*args, batch_size=batch_size, **kwargs)


class FixedHP:
    """Hyperparamètres imposés : mêmes méthodes que kt.HyperParameters, mais renvoie les valeurs données."""
    def __init__(self, values):
        self.values = dict(values)

    def Int(self, name, *a, **k): return self.values[name]
    def Choice(self, name, *a, **k): return self.values[name]
    def Float(self, name, *a, **k): return self.values[name]
    def get(self, name): return self.values[name]


def manual_values(arch, m):
    """Réglages manuels de l'interface → valeurs des noms d'hyperparamètres utilisés par les builders."""
    d = float(m.get("dropout", 0.2)); lr = float(m.get("lr", 1e-3)); bs = int(m.get("batch", 32))
    v = {"lr": lr, "batch_size": bs}
    for i in range(3):
        v[f"dropout_{i}"] = d
    if arch == "dense":
        v.update({"n_layers": 2, "units_0": 128, "units_1": 64, "units_2": 32})
    elif arch == "cnn":
        v.update({"n_conv_blocks": 2, "filters_0": 32, "filters_1": 64, "filters_2": 64,
                  "kernel_0": 5, "kernel_1": 5, "kernel_2": 5, "dense_units": 64})
    elif arch == "lstm":
        v.update({"n_lstm_layers": 1, "lstm_units_0": 64, "lstm_units_1": 32, "lstm_units_2": 32, "dense_units": 32})
    else:
        raise ValueError(f"architecture inconnue : {arch}")
    return v


class _TrialCounter(kt.Hyperband):
    """Hyperband + rappel à la fin de chaque essai (pour la barre de progression)."""
    def __init__(self, *a, on_trial=None, **k):
        super().__init__(*a, **k)
        self._on_trial, self._n_trials = on_trial, 0

    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        self._n_trials += 1
        if self._on_trial:
            try:
                self._on_trial(self._n_trials)
            except Exception:
                pass


class _Progress(keras.callbacks.Callback):
    def __init__(self, on_epoch, cancel):
        super().__init__()
        self._on_epoch, self._cancel = on_epoch, cancel

    def on_epoch_end(self, epoch, logs=None):
        if self._on_epoch:
            try:
                self._on_epoch(epoch + 1, logs or {})
            except Exception:
                pass
        if self._cancel and self._cancel():
            self.model.stop_training = True


class KerasEngine:
    def __init__(self, tuner_dir, cancel=None):
        self.tuner_dir = tuner_dir
        self.cancel = cancel

    # ── tuning : Hyperband, objectif val_accuracy, EarlyStopping(val_loss) — comme tune_hyperparameters() ──
    def tune(self, arch, input_shape, n_classes, Xtr, ytr, Xval, yval, hb, tag, on_trial=None):
        max_epochs = int(hb.get("max_epochs", MAX_EPOCHS))
        hypermodel = GestureHyperModel(arch, input_shape, n_classes)
        tuner = _TrialCounter(
            hypermodel,
            objective="val_accuracy",
            max_epochs=max_epochs,
            factor=int(hb.get("factor", HYPERBAND_FACTOR)),
            hyperband_iterations=int(hb.get("iterations", HYPERBAND_ITERATIONS)),
            directory=self.tuner_dir,
            project_name=f"{tag}_{arch}",
            overwrite=True,
            on_trial=on_trial,
        )
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=int(hb.get("patience", EARLY_STOPPING_PATIENCE)), restore_best_weights=True)
        tuner.search(Xtr, ytr, validation_data=(Xval, yval), epochs=max_epochs,
                     callbacks=[early_stop, _Progress(None, self.cancel)], verbose=0)
        return tuner.get_best_hyperparameters(1)[0]

    def manual_hp(self, arch, manual):
        full = (manual.get("hp") or {}).get(arch)          # hyperparamètres complets de l'interface (couches, unités, …)
        return FixedHP(full if full else manual_values(arch, manual))

    def fixed_hp(self, values):
        """Hyperparamètres copiés d'un run précédent (mêmes noms que hp.values)."""
        return FixedHP(values)

    def hp_values(self, hp):
        out = {}
        for k, v in dict(hp.values).items():
            out[k] = float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v
        return out

    # ── entraînement d'un pli — comme train_and_eval_fold() ──
    def train_eval(self, arch, best_hp, input_shape, n_classes, Xtr, ytr, Xval, yval, Xte, epochs, patience, on_epoch=None):
        model = ARCH_BUILDERS[arch](best_hp, input_shape, n_classes)
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True)
        model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=epochs,
                  batch_size=best_hp.get("batch_size"), callbacks=[early_stop, _Progress(on_epoch, self.cancel)], verbose=0)
        test_prob = model.predict(Xte, verbose=0)
        return model, test_prob

    def save(self, model, path, scaler=None):
        """Sauvegarde AUTONOME : la normalisation du pli (StandardScaler du train) est intégrée comme première couche,
        le fichier prend donc des données brutes. Retourne True si c'est le cas ; sinon sauvegarde simple (False)
        et l'appelant garde moyenne / écart-type à côté."""
        if scaler is not None:
            try:
                norm = layers.Normalization(axis=-1, mean=np.asarray(scaler.mean_, dtype="float32"),
                                            variance=np.asarray(scaler.scale_, dtype="float32") ** 2)
                inp = keras.Input(shape=tuple(model.input_shape[1:]))
                keras.Model(inp, model(norm(inp))).save(path)
                return True
            except Exception:
                pass
        model.save(path)
        return False

    def free(self, model):
        del model
        keras.backend.clear_session()
