"""
model_template.py — script modèle : entraîne TON modèle sur le dataset TapStrap et le prépare pour l'import dans la page « Classifieur ».

Deux façons d'utiliser la page :
  1. Tu utilises nos classifieurs (Dense / CNN / LSTM) : rien à faire, tout est dans la page.
  2. Tu fais comme tu veux (ce script) et tu importes ton modèle dans « Modèles › Importer un modèle ».

Ce script fait, dans cet ordre :
  A. charge les répétitions nettoyées exactement comme la page (mêmes fichiers, mêmes colonnes, T pas de temps) ;
  B. sépare train / validation / test par PARTICIPANT (le test n'est jamais vu à l'entraînement) ;
  C. normalise : z-score par colonne, moyenne et écart-type calculés sur le TRAIN seulement  ← la règle de la page ;
  D. appelle TON modèle (la seule partie à modifier : build_model) ;
  E. évalue sur le participant de test ;
  F. sauvegarde  <NOM>.keras (ou .joblib)  +  <NOM>_normalization.json  → à importer ensemble dans la page.

Usage :   python model_template.py "chemin/vers/data_clean"
Dépendances : numpy pandas scikit-learn  (+ tensorflow si FRAMEWORK = "keras")
"""
import os, re, sys, glob, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ═════════════════════════════════════════════════════════════
#  RÉGLAGES
# ═════════════════════════════════════════════════════════════
FRAMEWORK = "keras"                  # "keras" (réseau de neurones) ou "sklearn" (modèle classique sur la série aplatie)
NAME = "mon_modele"                  # nom des fichiers produits
T = 100                              # pas de temps : chaque répétition est ramenée à T points (interpolation linéaire)
EXCLUDE_POINTS = ["TAP1_Accel"]      # points de capteur à ne PAS utiliser (comme la page : TAP1_Accel exclu par défaut)
TEST_USER = "P01"                    # participant gardé pour le test ; None = tout sert à l'entraînement (pas d'évaluation)
N_VAL_USERS = 2                      # participants de validation (tirés au sort parmi les autres)
SEED = 42
EPOCHS, BATCH = 30, 32               # pour Keras


# ═════════════════════════════════════════════════════════════
#  A. CHARGEMENT  (identique à la page — ne pas modifier si tu veux des résultats comparables)
# ═════════════════════════════════════════════════════════════
FILENAME = re.compile(r"^P(\d+)_S(\d+)_Gesture_(\d+)_rep(\d+)_(\d+)_(\d+)_Clean\.csv$", re.I)
META_COLS = {"Participant", "Gesture", "Repetition", "Timestamp_LSL"}


def point_of(col):
    return re.sub(r"_[XYZ]$", "", col)             # 'TAP1_Thumb_X' -> 'TAP1_Thumb'


def resample(arr, n):
    src, dst = np.linspace(0, 1, arr.shape[0]), np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, arr[:, j]) for j in range(arr.shape[1])], axis=1).astype(np.float32)


def load_dataset(root):
    files = {}
    for gd in glob.glob(os.path.join(root, "Gesture_*")):
        for r, _d, fs in os.walk(gd):
            for f in fs:
                m = FILENAME.match(f)
                if m:
                    files[(f"P{int(m[1]):02d}", f"Gesture_{int(m[3]):02d}", int(m[2]) * 100 + int(m[4]))] = os.path.join(r, f)
    if not files:
        sys.exit("Aucun fichier *_Clean.csv trouvé dans " + root)
    first = pd.read_csv(files[sorted(files)[0]], nrows=3)
    cols = [c for c in first.columns if c not in META_COLS and first[c].dtype.kind in "fi" and point_of(c) not in EXCLUDE_POINTS]
    X, y, users = [], [], []
    for (u, g, _i), path in sorted(files.items()):
        X.append(resample(pd.read_csv(path)[cols].values.astype(np.float32), T))
        y.append(g); users.append(u)
    return np.stack(X), np.array(y), np.array(users), cols


# ═════════════════════════════════════════════════════════════
#  D. TON MODÈLE  ← la seule partie à modifier
# ═════════════════════════════════════════════════════════════
def build_model(input_shape, n_classes):
    """input_shape = (T, nombre_de_colonnes). Retourne un modèle Keras compilé (ou, en mode sklearn, un estimateur)."""
    if FRAMEWORK == "keras":
        from tensorflow import keras
        from tensorflow.keras import layers
        m = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(32, 5, activation="relu", padding="same"),     # ← remplace par ton architecture
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 5, activation="relu", padding="same"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dense(n_classes, activation="softmax"),               # la dernière couche doit donner n_classes probabilités
        ])
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return m
    from sklearn.ensemble import RandomForestClassifier                  # ← ou n'importe quel modèle scikit-learn
    return RandomForestClassifier(n_estimators=200, random_state=SEED)


# ═════════════════════════════════════════════════════════════
#  PROGRAMME
# ═════════════════════════════════════════════════════════════
def main(root):
    X, y, users, cols = load_dataset(root)
    classes = sorted(set(y))
    y_id = np.array([classes.index(g) for g in y])
    print(f"{len(X)} répétitions · {len(set(users))} participants · {len(classes)} gestes · {len(cols)} colonnes · T={T}")

    # B. séparation par participant
    others = sorted(str(u) for u in set(users) - {TEST_USER})
    rng = np.random.RandomState(SEED); rng.shuffle(others)
    val_users, train_users = others[:N_VAL_USERS], others[N_VAL_USERS:]
    tr, va = np.isin(users, train_users), np.isin(users, val_users)
    te = users == TEST_USER if TEST_USER else np.zeros(len(users), bool)
    print(f"train {sorted(train_users)} · val {sorted(val_users)} · test {TEST_USER}")

    # C. normalisation : z-score par colonne, statistiques du TRAIN seulement
    sc = StandardScaler().fit(X[tr].reshape(-1, X.shape[2]))
    norm = lambda A: sc.transform(A.reshape(-1, A.shape[2])).reshape(A.shape).astype(np.float32)
    Xtr, Xva, Xte = norm(X[tr]), norm(X[va]), norm(X[te]) if te.any() else None

    # D/E. entraînement + évaluation
    model = build_model((T, len(cols)), len(classes))
    if FRAMEWORK == "keras":
        from tensorflow import keras
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(Xtr, y_id[tr], validation_data=(Xva, y_id[va]), epochs=EPOCHS, batch_size=BATCH, callbacks=[stop], verbose=2)
        predict = lambda A: model.predict(A, verbose=0).argmax(1)
    else:
        model.fit(Xtr.reshape(len(Xtr), -1), y_id[tr])
        predict = lambda A: model.predict(A.reshape(len(A), -1))
    if Xte is not None:
        print(f"\nAccuracy sur {TEST_USER} (jamais vu à l'entraînement) : {(predict(Xte) == y_id[te]).mean():.3f}")

    # F. sauvegarde : modèle + fichier de normalisation
    if FRAMEWORK == "keras":
        model.save(NAME + ".keras"); model_file = NAME + ".keras"
    else:
        import joblib; joblib.dump(model, NAME + ".joblib"); model_file = NAME + ".joblib"
    with open(NAME + "_normalization.json", "w", encoding="utf-8") as f:
        json.dump({"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "columns": cols, "classes": classes, "T": T,
                   "note": "z-score par colonne, calculé sur le train du script model_template.py"}, f)
    print(f"\nFichiers prêts à importer dans la page : {model_file}  +  {NAME}_normalization.json")
    print("→ Classifieur › Modèles › Importer un modèle : choisis le modèle, normalisation « Z-score », fichier de normalisation = le .json.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('Usage : python model_template.py "chemin/vers/data_clean"')
    main(sys.argv[1])
