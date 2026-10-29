"""
Expérience TapStrap — Orchestrateur principal (v3)
N appareils Tap décrits par un registre (TapStrap2 / TapXR) + LSL + CSV backup + Flask UI

Différence avec la v2 : les deux slots TAP1/TAP2 ne sont plus écrits en dur. Les appareils, leur
modèle, leur position sur le corps et les flux qu'ils produisent viennent d'un registre de slots, modifiable
depuis l'examinateur (type + nom Bluetooth + « Ajouter ») et enregistré dans devices.json. Le type et le nom
Bluetooth de chaque appareil sont saisis à la main : rien n'est deviné, et un appareil dont le nom n'est dans aucun
slot est ignoré. Avec le registre par défaut (TapStrap2 = TAP1, TapXR = TAP2, noms à saisir), les noms de flux et
de fichiers sont ceux de la v2.
"""

from flask import Flask, render_template, jsonify, request, send_file, make_response
from tapsdk import TapSDK, TapInputMode
from pylsl import StreamInfo, StreamOutlet, local_clock
import csv, os, shutil, threading, time, json, queue, glob

from datetime import datetime
from collections import deque
import dataset_tools as dt   # inventaire / contrôle / déplacement du dataset brut
import preprocessing as pp   # chaîne de prétraitement (régression d'horloge → dataset aligné)
import classifier_v2 as clf     # classifieurs (Dense / LSTM / CNN), LORO / LOUO, modèles pré-entraînés

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════
#  CONFIG  ← adapte ici
# ═══════════════════════════════════════════════════════════
# Chaque geste : nom, description, fichier vidéo dans static/videos/
GESTURE_DEFS = [
    

  {
    "name": "Gesture 01",
    "video": "gesture_1.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras posé sur l'accoudoir, main dans le prolongement de l'avant-bras, paume vers le bas, doigts détendus et légèrement écartés.",
    "start_en": "Arm resting on the armrest, hand in line with the forearm, palm facing down, fingers relaxed and slightly apart.",
    "desc_fr": "Tous les doigts se referment simultanément vers la paume jusqu'à former un poing serré.",
    "desc_en": "All fingers curl inward simultaneously toward the palm until a tight fist is formed."
  },
  {
    "name": "Gesture 02",
    "video": "gesture_2.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras le long du corps, main dans le prolongement de l'avant-bras, paume vers soi, doigts refermés.",
    "start_en": "Arm along the body, hand in line with the forearm, palm facing toward oneself, fingers closed.",
    "desc_fr": "Les cinq doigts s'étendent complètement et s'écartent les uns des autres.",
    "desc_en": "All five fingers extend fully and spread apart from each other."
  },
  {
    "name": "Gesture 03",
    "video": "gesture_3.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras posé sur l'accoudoir, main dans le prolongement de l'avant-bras, paume vers le bas, doigts détendus.",
    "start_en": "Arm resting on the armrest, hand in line with the forearm, palm facing down, fingers relaxed.",
    "desc_fr": "L'index est placé contre la base du pouce ; l'index glisse le long du pouce, puis est relâché et s'étend complètement.",
    "desc_en": "The index finger is placed at the base of the thumb; the index slides along the thumb, then is released and fully extends."
  },
  {
    "name": "Gesture 04",
    "video": "gesture_4.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras le long du corps, main détendue, doigts dans le prolongement naturel de l'avant-bras.",
    "start_en": "Arm along the body, hand relaxed, fingers in the natural extension of the forearm.",
    "desc_fr": "L'index et le majeur s'étendent vers le haut ; l'annulaire, l'auriculaire et le pouce restent repliés.",
    "desc_en": "The index and middle fingers extend upward; the ring finger, pinky, and thumb remain closed."
  },
  {
    "name": "Gesture 05",
    "video": "gesture_5.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras posé sur l'accoudoir, paume orientée face à l'écran, pouce et index écartés en forme de L, majeur aligné avec l'index.",
    "start_en": "Arm resting on the armrest, palm facing the screen, thumb and index spread open in an L-shape, middle finger aligned with the index.",
    "desc_fr": "Le pouce, l'index et le majeur (index et majeur collés) convergent l'un vers l'autre en un mouvement de pincement, comme pour comprimer un contenu sur un écran.",
    "desc_en": "The thumb, index, and middle finger (index and middle kept together) converge toward each other in a pinching motion, as if compressing content on a screen."
  },
  {
    "name": "Gesture 06",
    "video": "gesture_6.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras posé sur l'accoudoir, paume orientée face à l'écran, pouce, index et majeur rapprochés en pince (index et majeur collés).",
    "start_en": "Arm resting on the armrest, palm facing the screen, thumb, index, and middle finger pinched together (index and middle kept together).",
    "desc_fr": "Le pouce, l'index et le majeur (index et majeur collés) s'écartent vers l'extérieur en un mouvement d'ouverture fluide, imitant le geste de zoom par pincement.",
    "desc_en": "The thumb, index, and middle finger (index and middle kept together) spread apart in a smooth outward motion, mimicking a pinch-to-zoom gesture."
  },
  {
    "name": "Gesture 07",
    "video": "gesture_7.mp4",
    "limb": "poignet",
    "limb_fr": "Poignet",
    "limb_en": "Wrist",
    "start_fr": "Bras posé sur l'accoudoir, avant-bras à plat, paume vers le bas, doigts détendus.",
    "start_en": "Arm resting on the armrest, forearm flat, palm facing down, fingers relaxed.",
    "desc_fr": "La main se fléchit vers le haut au niveau du poignet, soulevant le dos de la main en direction de l'avant-bras. L'avant-bras reste immobile.",
    "desc_en": "The hand bends upward at the wrist, raising the back of the hand toward the forearm. The forearm remains stationary."
  },
  {
    "name": "Gesture 08",
    "video": "gesture_8.mp4",
    "limb": "poignet",
    "limb_fr": "Poignet",
    "limb_en": "Wrist",
    "start_fr": "Bras posé sur l'accoudoir, avant-bras à plat, paume vers le bas, doigts détendus.",
    "start_en": "Arm resting on the armrest, forearm flat, palm facing down, fingers relaxed.",
    "desc_fr": "La main se fléchit vers le bas au niveau du poignet, rapprochant la paume de l'avant-bras. L'avant-bras reste immobile.",
    "desc_en": "The hand bends downward at the wrist, moving the palm toward the forearm. The forearm remains stationary."
  },
  {
    "name": "Gesture 09",
    "video": "gesture_9.mp4",
    "limb": "poignet",
    "limb_fr": "Poignet",
    "limb_en": "Wrist",
    "start_fr": "Bras posé sur l'accoudoir, avant-bras à plat, paume orientée vers la gauche (pouce vers le haut).",
    "start_en": "Arm resting on the armrest, forearm flat, palm facing left (thumb up).",
    "desc_fr": "Le poignet dévie latéralement vers l'extérieur dans le plan horizontal, éloignant la main du milieu du corps.",
    "desc_en": "The wrist deviates laterally outward in the horizontal plane, displacing the hand away from the body midline."
  },
  {
    "name": "Gesture 10",
    "video": "gesture_10.mp4",
    "limb": "poignet",
    "limb_fr": "Poignet",
    "limb_en": "Wrist",
    "start_fr": "Bras posé sur l'accoudoir, avant-bras à plat, paume orientée vers la gauche (pouce vers le haut).",
    "start_en": "Arm resting on the armrest, forearm flat, palm facing left (thumb up).",
    "desc_fr": "Le poignet dévie médialement dans le plan horizontal, rapprochant la main du milieu du corps.",
    "desc_en": "The wrist deviates medially in the horizontal plane, displacing the hand toward the body midline."
  },
  {
    "name": "Gesture 11",
    "video": "gesture_11.mp4",
    "limb": "poignet",
    "limb_fr": "Poignet",
    "limb_en": "Wrist",
    "start_fr": "Bras posé sur l'accoudoir, avant-bras à plat, paume orientée vers le bas, doigts légèrement fléchis.",
    "start_en": "Arm resting on the armrest, forearm flat, palm facing down , fingers slightly curled.",
    "desc_fr": "L'avant-bras effectue une rotation de façon à ce que la paume se retourne vers le haut. Le bras reste collé à l'accoudoir pendant tout le mouvement.",
    "desc_en": "The forearm rotates so that the palm turns upward."
  },
  {
    "name": "Gesture 12",
    "video": "gesture_12.mp4",
    "limb": "poignet",
    "limb_fr": "Poignet",
    "limb_en": "Wrist",
    "start_fr": "Bras posé sur l'accoudoir, coude fléchi, avant-bras vertical, main dans le prolongement, paume vers le haut.",
    "start_en": "Arm resting on the armrest, elbow bent, forearm vertical, hand in line with the forearm, palm facing up.",
    "desc_fr": "L'avant-bras effectue une rotation, retournant la paume vers le bas. Le bras reste collé à l'accoudoir pendant tout le mouvement.",
    "desc_en": "The forearm rotates, turning the palm downward. The arm remains in contact with the armrest throughout the movement."
  },
  {
    "name": "Gesture 13",
    "video": "gesture_13.mp4",
    "limb": "coude",
    "limb_fr": "Coude",
    "limb_en": "Elbow",
    "start_fr": "Bras posé sur l'accoudoir, coude fléchi, avant-bras vertical, main en forme de poing.",
    "start_en": "Arm resting on the armrest, elbow bent, forearm vertical, hand in a fist.",
    "desc_fr": "L'avant-bras s'étend lentement jusqu'à ce que le coude soit complètement déplié.",
    "desc_en": "The forearm slowly straightens until the elbow is fully extended."
  },
  {
    "name": "Gesture 14",
    "video": "gesture_14.mp4",
    "limb": "coude",
    "limb_fr": "Coude",
    "limb_en": "Elbow",
    "duration": 5,
    "start_fr": "Bras posé sur l'accoudoir, coude à 90°, avant-bras horizontal, paume orientée vers soi.",
    "start_en": "Arm resting on the armrest, elbow at 90°, forearm horizontal, palm facing toward oneself.",
    "desc_fr": "Flexion-extension répétée du coude uniquement (2 fois) ; l'avant-bras restant horizontal et la paume orientée vers l'intérieur. On amène le bras vers soi, puis on revient en position de départ, puis on ramène le bras vers soi.",
    "desc_en": "Repeated flexion-extension of the elbow only (2 times), keeping the forearm horizontal and the palm facing inward. The arm is pulled toward oneself, then returned to the starting position, then pulled toward oneself again."
  },
  {
    "name": "Gesture 15",
    "video": "gesture_15.mp4",
    "limb": "coude",
    "limb_fr": "Coude",
    "limb_en": "Elbow",
    "start_fr": "Avant-bras entièrement étendu vers la droite, main dans le prolongement, paume vers le bas.",
    "start_en": "Forearm fully extended to the right, hand in line with the forearm, palm facing down.",
    "desc_fr": "En gardant le bras supérieur immobile, le coude se fléchit pour ramener l'avant-bras vers le corps, s'arrêtant quand le bras est à 90°.",
    "desc_en": "Keeping the upper arm still, the elbow bends to bring the forearm inward toward the body, stopping when the arm reaches 90°."
  },
  {
    "name": "Gesture 16",
    "video": "gesture_16.mp4",
    "limb": "coude",
    "limb_fr": "Coude",
    "limb_en": "Elbow",
    "duration": 5,
    "start_fr": "Coude sur l'accoudoir, bras fléchi vers le haut, doigts formant un angle droit par rapport à la main.",
    "start_en": "Elbow on the armrest, arm bent upward, fingers forming a right angle relative to the hand.",
    "desc_fr": "Tracé d'un Z en trois temps : trait 1 : mouvement latéral de l'épaule formant la barre horizontale du haut ; trait 2 : flexion/extension coordonnées du coude traçant la diagonale ; trait 3 : mouvement latéral du coude formant la barre horizontale du bas.",
    "desc_en": "Tracing a Z in three strokes: stroke 1 — shoulder-driven lateral motion forms the top horizontal bar; stroke 2 — coordinated elbow flexion/extension traces the diagonal; stroke 3 — elbow-driven lateral motion forms the bottom horizontal bar."
  },
  {
    "name": "Gesture 17",
    "video": "gesture_17.mp4",
    "limb": "coude",
    "limb_fr": "Coude",
    "limb_en": "Elbow",
    "start_fr": "Bras posé sur l'accoudoir, paume vers le bas.",
    "start_en": "Arm resting on the armrest, palm facing down.",
    "desc_fr": "L'avant-bras se lève pour se poser contre le bras, comme pour lever la main, mais le coude reste sur l'accoudoir.",
    "desc_en": "The forearm lifts up against the upper arm, as if raising the hand, but the elbow remains on the armrest."
  },
  {
    "name": "Gesture 18",
    "video": "gesture_18.mp4",
    "limb": "coude",
    "limb_fr": "Coude",
    "limb_en": "Elbow",
    "start_fr": "Sans accoudoir, bras posé comme s'il était sur l'accoudoir, paume vers le bas.",
    "start_en": "Without armrest, arm held as if resting on an armrest, palm facing down.",
    "desc_fr": "L'avant-bras se dirige vers le bas, comme si on laissait pendre son bras. Le bras supérieur reste immobile.",
    "desc_en": "The forearm moves downward, as if letting the arm hang. The upper arm remains stationary."
  },
  {
    "name": "Gesture 19",
    "video": "gesture_19.mp4",
    "limb": "epaule",
    "limb_fr": "Épaule",
    "limb_en": "Shoulder",
    "start_fr": "Bras le long du corps, main détendue, doigts dans le prolongement naturel de l'avant-bras.",
    "start_en": "Arm along the body, hand relaxed, fingers in the natural extension of the forearm.",
    "desc_fr": "Le bras se soulève vers l'avant, depuis la position basse jusqu'à l'élévation maximale.",
    "desc_en": "The arm lifts forward, from the lowered position to maximum elevation."
  },
  {
    "name": "Gesture 20",
    "video": "gesture_20.mp4",
    "limb": "epaule",
    "limb_fr": "Épaule",
    "limb_en": "Shoulder",
    "start_fr": "Bras le long du corps, main détendue, doigts dans le prolongement naturel de l'avant-bras.",
    "start_en": "Arm along the body, hand relaxed, fingers in the natural extension of the forearm.",
    "desc_fr": "Le bras s'élève latéralement, s'éloignant du milieu du corps, jusqu'à l'élévation maximale.",
    "desc_en": "The arm raises laterally, moving away from the body midline, up to maximum elevation."
  },
  {
    "name": "Gesture 21",
    "video": "gesture_21.mp4",
    "limb": "epaule",
    "limb_fr": "Épaule",
    "limb_en": "Shoulder",
    "start_fr": "Bras étendu, paume vers l'écran.",
    "start_en": "Arm extended, palm facing the screen.",
    "desc_fr": "Le bras balaye horizontalement vers la droite en un seul mouvement, sans arriver à l'amplitude maximale.",
    "desc_en": "The arm sweeps horizontally to the right in a single motion, without reaching maximum range."
  },
  {
    "name": "Gesture 22",
    "video": "gesture_22.mp4",
    "limb": "epaule",
    "limb_fr": "Épaule",
    "limb_en": "Shoulder",
    "start_fr": "Bras étendu horizontalement sur le côté, main dans le prolongement, doigts détendus.",
    "start_en": "Arm extended horizontally to the side, hand in line with the forearm, fingers relaxed.",
    "desc_fr": "Le bras descend en arc de cercle contrôlé dans le plan frontal jusqu'à se retrouver le long du corps.",
    "desc_en": "The arm sweeps downward in a controlled arc through the frontal plane until it hangs straight down along the body."
  },
  {
    "name": "Gesture 23",
    "video": "gesture_23.mp4",
    "limb": "epaule",
    "limb_fr": "Épaule",
    "limb_en": "Shoulder",
    "start_fr": "Bras étendu, paume vers l'écran.",
    "start_en": "Arm extended, palm facing the screen.",
    "desc_fr": "Le bras balaye horizontalement vers la gauche en un seul mouvement, sans arriver à l'amplitude maximale.",
    "desc_en": "The arm sweeps horizontally to the left in a single motion, without reaching maximum range."
  },
  {
    "name": "Gesture 24",
    "video": "gesture_24.mp4",
    "limb": "epaule",
    "limb_fr": "Épaule",
    "limb_en": "Shoulder",
    "start_fr": "Bras étendu, poing fermé.",
    "start_en": "Arm extended, fist closed.",
    "duration": 5,
    "desc_fr": "Le bras trace un huit couché continu : depuis le centre, boucle vers la droite, retour par le centre, boucle vers la gauche, puis retour au centre.",
    "desc_en": "The arm traces a continuous sideways figure-of-eight: from center, loop right, cross back through center, loop left, and return to center."
  },
{
    "name": "Gesture 25",
    "video": "gesture_25.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Coude posé sur l'accoudoir, l'avant-bras vers le haut, paume vers le coté, doigts detendus  .",
    "start_en": "Elbow resting on the armrest, forearm pointing upward, palm on the side, fingers relaxed.",
    "desc_fr": "Les doigts se referment en poing pendant que le pouce se lève vers le haut (geste « pouce en l'air »).",
    "desc_en": "The fingers close into a fist while the thumb rises upward (thumbs-up gesture)."
  },
  {
    "name": "Gesture 26",
    "video": "gesture_26.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras posé sur l'accoudoir, main dans le prolongement de l'avant-bras, paume vers le bas, doigts refermés.",
    "start_en": "Arm resting on the armrest, hand in line with the forearm, palm facing down, fingers closed.",
    "desc_fr": "L'index s'étend vers le haut comme pour pointer quelque chose, les autres doigts restent repliés.",
    "desc_en": "The index finger extends upward as if pointing at something, the other fingers remain folded."
  },
  {
    "name": "Gesture 27",
    "video": "gesture_27.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras posé sur l'accoudoir, main dans le prolongement de l'avant-bras, paume vers le bas, doigts refermés.",
    "start_en": "Arm resting on the armrest, hand in line with the forearm, palm facing down, fingers closed.",
    "desc_fr": "L'index et le majeur s'étendent et s'écartent en forme de V, les autres doigts et le pouce restent repliés.",
    "desc_en": "The index and middle fingers extend and spread apart in a V shape, the other fingers and thumb remain folded."
  },
  {
    "name": "Gesture 28",
    "video": "gesture_28.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Coude posé sur l'accoudoir, l'avant-bras vers le haut, paume vers l'écran, doigts refermés.",
    "start_en": "Elbow resting on the armrest, forearm pointing upward, palm facing the screen, fingers closed.",
    "desc_fr": "Le pouce et l'index se rejoignent en formant un cercle, les autres doigts s'étendent naturellement.",
    "desc_en": "The thumb and index finger come together to form a circle, the other fingers extend naturally."
  },
  {
    "name": "Gesture 29",
    "video": "gesture_29.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Coude posé sur l'accoudoir, l'avant-bras vers le haut, paume vers l'écran, doigts refermés.",
    "start_en": "Elbow resting on the armrest, forearm pointing upward, palm facing the screen, fingers closed.",
    "desc_fr": "L'auriculaire s'étend vers le haut, les autres doigts restent repliés.",
    "desc_en": "The pinky finger extends upward, the other fingers remain folded."
  },
  {
    "name": "Gesture 30",
    "video": "gesture_30.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Coude posé sur l'accoudoir, l'avant-bras vers le haut, paume vers l'écran, doigts refermés.",
    "start_en": "Elbow resting on the armrest, forearm pointing upward, palm facing the screen, fingers closed.",
    "desc_fr": "Les quatre doigts se fléchissent simultanément en griffes, les pointes se rapprochant de la paume sans former un poing complet.",
    "desc_en": "All four fingers curl simultaneously into a claw shape, fingertips approaching the palm without forming a full fist."
  },
  {
    "name": "Gesture 31",
    "video": "gesture_31.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Coude posé sur l'accoudoir, l'avant-bras vers le haut, paume vers l'écran, doigts refermés.",
    "start_en": "Elbow resting on the armrest, forearm pointing upward, palm facing the screen, fingers closed.",
    "desc_fr": "Tous les doigts s'étendent complètement.",
    "desc_en": "All fingers extend fully."
  },
  {
    "name": "Gesture 32",
    "video": "gesture_32.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras posé sur l'accoudoir, l'avant-bras vers le haut, paume vers l'écran, doigts en position comme pour faire un 2.",
    "start_en": "Arm resting on the armrest, hand in line with the forearm, palm facing the screen, fingers positioned as if making the number 2.",
    "desc_fr": "L'index et le majeur s'écartent simultanément.",
    "desc_en": "The index and middle fingers spread apart simultaneously."
  },
  {
    "name": "Gesture 33",
    "video": "gesture_33.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Coude posé sur l'accoudoir, l'avant-bras vers le haut, paume vers l'écran, poing fermé.",
    "start_en": "Elbow resting on the armrest, forearm pointing upward, palm facing the screen, fist closed.",
    "desc_fr": "L'index passe par-dessus le majeur, les deux doigts restant étendus, les autres doigts restent repliés.",
    "desc_en": "The index finger crosses over the middle finger, both fingers remaining extended, the other fingers remain folded."
  },
  {
    "name": "Gesture 34",
    "video": "gesture_34.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "duration": 9,
    "start_fr": "Avant-bras reposant sur l'accoudoir, paume vers le haut.",
    "start_en": "Forearm resting on the armrest, palm facing upward.",
    "desc_fr": "Le pouce touche successivement chacun des doigts selon les 10 positions du score de Kapandji : du bout de l'index jusqu'à la base de l'auriculaire, en passant par les positions intermédiaires de chaque doigt.",
    "desc_en": "The thumb touches each finger successively according to the 10 positions of the Kapandji score: from the tip of the index finger to the base of the pinky, passing through intermediate positions of each finger."
  }
]

# GESTURE_DEFS ci-dessus = le jeu d'origine (« default »). D'autres jeux se créent depuis l'onglet
# « Jeux de gestes » de l'examinateur ; le jeu ACTIF remplace GESTURE_DEFS / GESTURES / GESTURE_MAP /
# LATIN_SQUARE / DATA_ROOT / CLEAN_ROOT (voir _apply_set plus bas).
BUILTIN_DEFS = GESTURE_DEFS
GESTURE_MAP = {g["name"]: g for g in GESTURE_DEFS}
GESTURES    = [g["name"] for g in GESTURE_DEFS]

REPETITIONS_PER_GESTURE = 5
COUNTDOWN_SECONDS       = 3
RECORDING_DURATION = 3
FINGER_SRATE            = 200.0
IMU_SRATE               = 208.0

# ═══════════════════════════════════════════════════════════
#  REGISTRE DES APPAREILS (v3)
# ═══════════════════════════════════════════════════════════
# Flux produits par chaque modèle d'appareil en mode raw.
DEVICE_MODELS = {
    "TapStrap2": ("finger", "imu"),   # 5 accéléromètres de doigts (15 canaux) + IMU du pouce (6 canaux)
    "TapXR":     ("imu",),            # IMU du bracelet uniquement (6 canaux)
}

# Le registre est une liste de slots. Un slot = un appareil attendu :
#   slot      nom du slot, utilisé dans les noms de flux LSL et de fichiers (lettres et chiffres seulement ; TAP1, TAP2… par défaut)
#   model     clé de DEVICE_MODELS
#   position  position sur le corps (texte libre, affiché dans l'examinateur)
#   name      nom Bluetooth de l'appareil à mettre dans ce slot (ex. « TapXR_A215310 »), tel qu'affiché par le système
# Le type ET le nom sont saisis à la main : rien n'est deviné. Un appareil dont le nom ne figure dans aucun slot est
# ignoré, et le scan est refusé tant qu'un slot n'a pas de nom. Le registre se modifie depuis l'examinateur
# (formulaire « type + nom Bluetooth + Ajouter », champ du nom de chaque slot, ✕) tant qu'aucun scan n'est en cours,
# et il est enregistré dans devices.json à côté de ce script. Les noms du registre par défaut sont à remplir.
DEFAULT_DEVICE_REGISTRY = [
    {"slot": "TAP1", "model": "TapStrap2", "position": "right hand",    "name": ""},   # nom Bluetooth à saisir
    {"slot": "TAP2", "model": "TapXR",     "position": "right forearm", "name": ""},   # nom Bluetooth à saisir
]
REGISTRY_FILE = os.path.join(BASE_DIR, "devices.json")

def _missing_names():
    """Slots dont le nom Bluetooth n'est pas renseigné."""
    return [d["slot"] for d in DEVICE_REGISTRY if not d["name"]]

def _normalize_registry(raw):
    """Valide une liste d'entrées {slot?, model, position?, name?} et renvoie le registre complet.
    Un slot absent reçoit le premier nom libre TAP1, TAP2… Les noms Bluetooth sont comparés sans tenir compte de la casse."""
    import re
    explicit = {str(d["slot"]) for d in raw if d.get("slot")}
    registry, used, names, nxt = [], set(), set(), 1
    for d in raw:
        model = d.get("model")
        if model not in DEVICE_MODELS:
            raise ValueError(f"Modèle inconnu « {model} » (connus : {', '.join(DEVICE_MODELS)})")
        slot = str(d["slot"]) if d.get("slot") else None
        if slot is None:
            while f"TAP{nxt}" in explicit or f"TAP{nxt}" in used:
                nxt += 1
            slot = f"TAP{nxt}"
        if not re.fullmatch(r"[A-Za-z0-9]+", slot):
            raise ValueError(f"Nom de slot invalide « {slot} » (lettres et chiffres seulement)")
        if slot in used:
            raise ValueError(f"Slot « {slot} » défini deux fois")
        name = str(d.get("name") or "").strip()
        if name and name.lower() in names:
            raise ValueError(f"Le nom Bluetooth « {name} » est déjà utilisé par un autre slot")
        used.add(slot)
        if name:
            names.add(name.lower())
        registry.append({"slot": slot, "model": model, "position": str(d.get("position") or ""),
                         "name": name, "streams": DEVICE_MODELS[model]})
    if not registry:
        raise ValueError("Il faut au moins un appareil")
    return registry

def _load_registry():
    # À chaque démarrage on repart de la config de base (TAP1 = TapStrap2, TAP2 = TapXR) :
    # les modifications faites dans l'examinateur ne durent que le temps de la session
    # et devices.json n'est plus lu.
    return _normalize_registry([dict(d) for d in DEFAULT_DEVICE_REGISTRY])

def _save_registry():
    pass   # volontairement vide : le registre n'est plus enregistré sur disque

# ── Tap déjà connectés (mémoire permanente, indépendante du registre de session) ──
KNOWN_TAPS_FILE = os.path.join(BASE_DIR, "known_taps.json")
_known_lock = threading.Lock()

def _known_taps():
    try:
        with open(KNOWN_TAPS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return [{"name": str(d["name"]), "model": str(d.get("model") or "")} for d in data if d.get("name")]
    except Exception:
        return []

def _remember_tap(name, model):
    """Mémorise un appareil qui vient de se connecter, pour le proposer dans la liste."""
    if not name:
        return
    with _known_lock:
        known = [d for d in _known_taps() if d["name"] != name]
        known.append({"name": name, "model": model})
        known.sort(key=lambda d: d["name"].lower())
        try:
            with open(KNOWN_TAPS_FILE, "w", encoding="utf-8") as f:
                json.dump(known, f, ensure_ascii=False, indent=2)
        except OSError as e:
            log(f"⚠ Impossible d'enregistrer known_taps.json : {e}")

# Seule configuration validée : un TapStrap2 et un TapXR. Une autre configuration est acceptée,
# mais signalée comme NON TESTÉE au démarrage, dans le log de l'examinateur et dans /api/state.
TESTED_CONFIGURATION = {"TapStrap2": 1, "TapXR": 1}
UNTESTED_CONFIGURATION = False
UNTESTED_NOTE = ""

DEVICE_REGISTRY = _load_registry()
SLOTS       = []
SLOT_INFO   = {}
STREAM_KEYS = []   # ex. finger_TAP1, imu_TAP1, imu_TAP2

def _refresh_registry_views():
    """Recalcule, en place, tout ce qui se déduit de DEVICE_REGISTRY : slots, flux, drapeau « non testé »."""
    global UNTESTED_CONFIGURATION, UNTESTED_NOTE
    SLOTS[:] = [d["slot"] for d in DEVICE_REGISTRY]
    SLOT_INFO.clear()
    SLOT_INFO.update({d["slot"]: d for d in DEVICE_REGISTRY})
    STREAM_KEYS[:] = [f"{st}_{d['slot']}" for d in DEVICE_REGISTRY for st in d["streams"]]
    counts = {}
    for d in DEVICE_REGISTRY:
        counts[d["model"]] = counts.get(d["model"], 0) + 1
    UNTESTED_CONFIGURATION = counts != TESTED_CONFIGURATION
    UNTESTED_NOTE = ("configuration d'appareils NON TESTÉE : seule 1 TapStrap2 + 1 TapXR a été validée "
                     f"(registre actuel : {', '.join(f'{n} {m}' for m, n in counts.items())}). "
                     "Le prétraitement, l'audit et le classifieur ne sont pas adaptés à d'autres configurations.")

_refresh_registry_views()
if UNTESTED_CONFIGURATION:
    print(f"[Registre] ⚠ {UNTESTED_NOTE}", flush=True)
MODEL_CHECK_DELAY = 10.0   # secondes après la connexion avant de vérifier que les données reçues correspondent au modèle déclaré


# ═══════════════════════════════════════════════════════════
#  LATIN SQUARE
# ═══════════════════════════════════════════════════════════
def build_random_latin_square(items):
    """Carré latin n×n aléatoire, valide PAR CONSTRUCTION (jamais d'échec ni de repli).

    On part du carré cyclique L[r][c] = (r + c) mod n puis on permute au hasard les lignes,
    les colonnes et les symboles : ces trois opérations conservent la propriété « chaque
    symbole exactement une fois par ligne et par colonne ».

    (L'ancienne version tirait les lignes au hasard avec repli cyclique : environ 29 % des
    générations pour n = 34 produisaient un carré invalide et faisaient échouer le démarrage.)

    Limite à connaître : un carré latin équilibre la POSITION de chaque geste, pas ses
    voisins. Pour équilibrer aussi quel geste précède quel autre (effets de report), il
    faudrait un carré de Williams (n pair)."""
    import random
    rng  = random.Random()                 # graine aléatoire
    n    = len(items)
    rows = list(range(n)); cols = list(range(n)); syms = list(items)
    rng.shuffle(rows); rng.shuffle(cols); rng.shuffle(syms)
    return [[syms[(r + c) % n] for c in cols] for r in rows]

def verify_latin_square(square):
    n = len(square); items = set(square[0])
    for row in square:
        if set(row) != items: return False
    for col in range(n):
        if {square[r][col] for r in range(n)} != items: return False
    return True

# ── Jeux de gestes : chacun a son carré latin et ses dossiers (voir gestureset.py) ──
import gestureset_customdata as gs
SETS = gs.GestureSets(BASE_DIR, BUILTIN_DEFS, REPETITIONS_PER_GESTURE, build_random_latin_square, verify_latin_square)
ACTIVE_SET = gs.DEFAULT_ID
ACTIVE_DATASET = gs.DEFAULT_DATASET
LATIN_SQUARE = []
# Arborescence : data/<jeu>/<dataset>/  et  data_clean/<jeu>/<dataset>/  (voir gestureset_customdata.py)
DATA_ROOT  = os.path.join(BASE_DIR, "data", gs.DEFAULT_ID, gs.DEFAULT_DATASET)
CLEAN_BASE = os.path.join(BASE_DIR, "data_clean", gs.DEFAULT_ID, gs.DEFAULT_DATASET)   # racine du dataset nettoyé
VIEW_SET = VIEW_DS = None     # dataset AFFICHÉ (vue d'ensemble, audit, prétraitement, classifieur) ; None = celui de l'enregistrement
VDATA, VGESTURES, VREPS = DATA_ROOT, [], 5   # bruts, gestes et répétitions du dataset affiché (mis à jour par _set_view)
CLEAN_VERSION = ""            # "" = racine ; sinon nom (slug) d'une version nommée : data_clean/<jeu>/<slug>
CLEAN_ROOT = CLEAN_BASE       # version nettoyée ACTIVE : tout le reste de l'app (stats, répétition, classifieur) la suit

def _apply_set(sid, ds=None):
    """Rend `sid` actif avec son dataset `ds` (par défaut : le dernier utilisé pour ce jeu) :
    gestes, répétitions, carré latin et racines de données."""
    global GESTURE_DEFS, GESTURE_MAP, GESTURES, REPETITIONS_PER_GESTURE, LATIN_SQUARE, DATA_ROOT, CLEAN_ROOT, CLEAN_BASE, CLEAN_VERSION, ACTIVE_SET, ACTIVE_DATASET
    st = SETS.get(sid, ds)
    GESTURE_DEFS = st["gestures"]
    GESTURE_MAP  = {g["name"]: g for g in GESTURE_DEFS}
    GESTURES     = [g["name"] for g in GESTURE_DEFS]
    REPETITIONS_PER_GESTURE = int(st["reps"])
    LATIN_SQUARE = st["latin"]
    DATA_ROOT, CLEAN_BASE = st["data_root"], st["clean_root"]
    CLEAN_VERSION, CLEAN_ROOT = "", CLEAN_BASE               # un autre dataset : retour à sa version par défaut
    ACTIVE_SET, ACTIVE_DATASET = sid, st["dataset"]
    print(f"[Gesture set] actif : {st['name']} ({sid}) · dataset {st['dataset']} — {len(GESTURES)} gestes × {REPETITIONS_PER_GESTURE} rép. — {DATA_ROOT}")
    _set_view()                                               # l'affichage suit de nouveau le dataset actif

def _set_view(sid=None, ds=None):
    """Choisit le dataset AFFICHÉ par les pages Dataset, Prétraitement et Classifieur. L'enregistrement, lui, reste sur le
    dataset actif (DATA_ROOT) : on peut nettoyer ou auditer un autre dataset sans toucher à l'acquisition.
    sid=None (ou le dataset actif) → l'affichage suit le dataset actif."""
    global VIEW_SET, VIEW_DS, VDATA, VGESTURES, VREPS, CLEAN_BASE, CLEAN_VERSION, CLEAN_ROOT
    if sid is None or (sid == ACTIVE_SET and (ds is None or ds == ACTIVE_DATASET)):
        VIEW_SET = VIEW_DS = None
        VDATA, VGESTURES, VREPS = DATA_ROOT, list(GESTURES), int(REPETITIONS_PER_GESTURE)
        base = SETS.roots(ACTIVE_SET, ACTIVE_DATASET)[1]
    else:
        st = SETS.get(sid, ds)                                # ValueError si le jeu ou le dataset n'existe pas
        VIEW_SET, VIEW_DS = sid, st["dataset"]
        VDATA, base = st["data_root"], st["clean_root"]
        VGESTURES, VREPS = [g["name"] for g in st["gestures"]], int(st["reps"])
    CLEAN_BASE, CLEAN_VERSION, CLEAN_ROOT = base, "", base

try:                                   # ancien format (data/P01…, data_clean/Gesture_01…) → data/default/principal/…
    for _line in SETS.migrate_legacy():
        print("[Migration]", _line)
except Exception as _e:
    print(f"[Migration] ⚠ échec : {_e} — rien n'a été supprimé, relance après avoir corrigé le problème")

try:
    _apply_set(SETS.active_id(), SETS.active_dataset())
except Exception as _e:
    print(f"[Gesture set] ⚠ jeu actif illisible ({_e}) — retour au jeu d'origine")
    _apply_set(gs.DEFAULT_ID)

# ═══════════════════════════════════════════════════════════
#  ÉTAT GLOBAL
# ═══════════════════════════════════════════════════════════
# Variables "rapides" sans lock pour le thread Bluetooth
FAST_SLOT_MAP = {}    # mapping identifier -> nom de slot du registre
FAST_RECORDING = False
FAST_FINGER_COUNT = 0
FAST_IMU_COUNT = 0
FAST_SLOT_COUNTS = {s: {"finger": 0, "imu": 0} for s in SLOTS}   # compteurs par slot (thread de traitement seul)
SLOT_SEEN = {s: {"finger": False, "imu": False} for s in SLOTS}   # types de paquets déjà reçus par slot depuis la connexion

state = {
    "participant_id":      1,
    "current_gesture_idx": 0,
    "current_repetition":  0,
    "recording":           False,
    "phase":               "idle",   # idle|ready|countdown|recording|done
    "gesture_order":       LATIN_SQUARE[0],
    "devices":             {d["slot"]: {"connected": False, "id": None, "name": None, "warning": None} for d in DEVICE_REGISTRY},
    "tap_scanning":        False,
    "finger_count":        0,
    "imu_count":           0,
    "countdown":           0,
    "video_file":          "gesture_demo.mp4",  # fallback si pas de vidéo par geste
    "log":                 deque(maxlen=50),
    "history":             [],
    "last_csv_paths":      {},
    "replay_requested":    False,
    "lsl_active":          False,
    "recording_start_time": None,
    "session_number":      1,
    "lsl_record_start":    0.0,
    "lsl_record_stop":     0.0,
    "lsl_countdown_start": 0.0,
    "lsl_cam_start_first_frame":    0.0,
    "lsl_cam_start_first_frame_hw": 0.0,
}
state_lock = threading.RLock()

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    with state_lock:
        state["log"].appendleft(f"[{ts}] {msg}")

# ═══════════════════════════════════════════════════════════
#  LSL — un outlet par flux de chaque appareil + 1 outlet markers
# ═══════════════════════════════════════════════════════════
# Les outlets sont créés une seule fois au tap_connect, d'après le registre,
# et poussent EN PERMANENCE — indépendamment du recording.
# LabRecorder choisit quand il capture. Les markers START/STOP
# permettent de couper le .xdf après coup si besoin.

lsl_markers            = None
lsl_lock               = threading.Lock()
lsl_outlets            = {}   # clé "<flux>_<slot>" -> outlet
lsl_keepalive          = {}   # références fortes : évite le garbage collection des outlets LSL
marker_log_path        = None
marker_log_lock        = threading.Lock()

def _make_outlet(stream_type, slot):
    """Crée l'outlet LSL d'un flux ('finger' ou 'imu') pour un slot."""
    if stream_type == "finger":
        info = StreamInfo(f"TapStrap_Finger_{slot}", "Finger", 15, FINGER_SRATE, "float32", f"tapstrap_finger_{slot.lower()}")
        chans = info.desc().append_child("channels")
        for finger in ["Thumb","Index","Middle","Ring","Pinky"]:
            for ax in ["X","Y","Z"]:
                ch = chans.append_child("channel")
                ch.append_child_value("label", f"{finger}_{ax}")
                ch.append_child_value("unit",  "normalized")
    else:
        info = StreamInfo(f"TapStrap_IMU_{slot}", "IMU", 6, IMU_SRATE, "float32", f"tapstrap_imu_{slot.lower()}")
        chans = info.desc().append_child("channels")
        for label, unit in zip(["Accel_X","Accel_Y","Accel_Z","Gyro_X","Gyro_Y","Gyro_Z"], ["m/s2"]*3+["rad/s"]*3):
            ch = chans.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit",  unit)
    return StreamOutlet(info, chunk_size=1, max_buffered=360)

def init_lsl_outlets():
    """Crée les streams LSL décrits par le registre, plus le stream de markers."""
    global lsl_markers

    print("[LSL] Création des outlets...", flush=True)

    outlets = {}
    for d in DEVICE_REGISTRY:
        for st in d["streams"]:
            key = f"{st}_{d['slot']}"
            outlets[key] = _make_outlet(st, d["slot"])
            print(f"[LSL] {key} OK", flush=True)

    # Markers
    info = StreamInfo("TapExp_Markers", "Markers", 1, 0, "string", "tapexp_markers")
    lsl_markers = StreamOutlet(info, chunk_size=1, max_buffered=360)
    print("[LSL] markers OK", flush=True)

    lsl_keepalive.clear()
    lsl_keepalive.update(outlets)
    with lsl_lock:
        lsl_outlets.clear()
        lsl_outlets.update(outlets)

    with state_lock:
        state["lsl_active"] = True
    log(f"LSL : {len(outlets) + 1} streams actifs — visible dans LabRecorder")
    print("[LSL] Tous les outlets créés ✓", flush=True)

def push_marker(marker_str, ts=None):
    marker_ts = local_clock() if ts is None else ts

    with lsl_lock:
        if lsl_markers:
            lsl_markers.push_sample([marker_str], marker_ts)

    with marker_log_lock:
        if marker_log_path:
            try:
                with open(marker_log_path, 'a', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow([f"{marker_ts:.6f}", marker_str])
            except Exception as e:
                print(f"[Marker Log Error] {e}")

# ═══════════════════════════════════════════════════════════
#  CSV BACKUP
# ═══════════════════════════════════════════════════════════
# On stocke tout en RAM et on écrit seulement à la fin
CSV_BUFFERS = {k: [] for k in STREAM_KEYS}   # une liste par flux du registre

FINGER_HEADER = ["Timestamp_LSL", "Timestamp_HW", "Type",
                 "Thumb_X","Thumb_Y","Thumb_Z",
                 "Index_X","Index_Y","Index_Z",
                 "Middle_X","Middle_Y","Middle_Z",
                 "Ring_X","Ring_Y","Ring_Z",
                 "Pinky_X","Pinky_Y","Pinky_Z"]
IMU_HEADER    = ["Timestamp_LSL", "Timestamp_HW", "Type",
                 "Accel_X","Accel_Y","Accel_Z",
                 "Gyro_X","Gyro_Y","Gyro_Z"]

def prepare_csv_buffers():
    """Génère les chemins mais n'ouvre pas les fichiers. Vide les buffers."""
    global cam_video_path, marker_log_path
    for k in CSV_BUFFERS:
        CSV_BUFFERS[k].clear()
    
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    pid     = state["participant_id"]
    gesture = state["gesture_order"][state["current_gesture_idx"]]
    rep     = state["current_repetition"]
    base    = os.path.dirname(os.path.abspath(__file__))
    session = state["session_number"]

    if str(pid).lower() == 'test':
        sdir   = os.path.join(base, "datatest", "test_P99")
        prefix = f"TEST_S{session:02d}_{gesture.replace(' ','_')}_rep{rep+1:02d}_{ts}"
    else:
        sdir   = os.path.join(DATA_ROOT, f"P{int(pid):02d}")
        prefix = f"P{int(pid):02d}_S{session:02d}_{gesture.replace(' ','_')}_rep{rep+1:02d}_{ts}"

    os.makedirs(sdir, exist_ok=True)
    
    paths   = {k: os.path.join(sdir, f"{prefix}_{k}.csv") for k in CSV_BUFFERS}
    marker_p = os.path.join(sdir, f"{prefix}_markers.csv")
    with open(marker_p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Timestamp_LSL", "Marker"])

    with state_lock:
        state["last_csv_paths"] = paths
    with marker_log_lock:
        marker_log_path = marker_p

    print(f"[Marker] fichier cree: {marker_p}", flush=True)
    log(f"Marker log cree: {os.path.basename(marker_p)}")
    log(f"Enregistrement prêt (CSV en différé) : {prefix}")

def write_and_close_csv_files():
    """Écrit tous les buffers en une seule fois dans les fichiers."""
    hdrs    = {k: (FINGER_HEADER if k.startswith("finger_") else IMU_HEADER) for k in CSV_BUFFERS}
    paths = state.get("last_csv_paths", {})
    
    for k, path in paths.items():
        if not path: continue
        buffer_data = CSV_BUFFERS[k]
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(hdrs[k])
                w.writerows(buffer_data)
        except Exception as e:
            print(f"[CSV Error] {e}")
    
    for k in CSV_BUFFERS:
        CSV_BUFFERS[k].clear()

# ═══════════════════════════════════════════════════════════
#  TAPSDK & QUEUE
# ═══════════════════════════════════════════════════════════
tap        = None
tap_thread = None

sensor_queue = queue.Queue()
queue_thread_active = False

def process_sensor_queue():
    """Thread en arrière-plan qui dépile la queue et écrit dans LSL/CSV pour ne pas bloquer le Bluetooth."""
    global queue_thread_active, FAST_FINGER_COUNT, FAST_IMU_COUNT
    while queue_thread_active:
        try:
            # Attend max 0.1s une donnée pour pouvoir sortir proprement de la boucle si on l'arrête
            data_item = sensor_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        lsl_ts, hardware_ts, slot, data_type, raw_points, recording_local = data_item
        
        sample = []
        for p in raw_points:
            sample += [float(p.x), float(p.y), float(p.z)] if hasattr(p,'x') else [0.,0.,0.]
            
        if 'Device' in data_type:
            while len(sample) < 15:
                sample.append(0.)

            if not SLOT_SEEN[slot]["finger"]:
                SLOT_SEEN[slot]["finger"] = True
                if "finger" not in SLOT_INFO[slot]["streams"]:
                    _warn_device(slot, f"paquets doigts reçus alors que le slot {slot} est déclaré {SLOT_INFO[slot]['model']} (sans flux doigts) : "
                                       f"ces données sont ignorées — corrige le modèle dans le registre")

            key = f"finger_{slot}"
            with lsl_lock:
                o = lsl_outlets.get(key)
                if o: o.push_sample(sample, lsl_ts)

            if recording_local:
                buf = CSV_BUFFERS.get(key)          # absent si le modèle de l'appareil n'a pas ce flux
                if buf is not None:
                    buf.append([lsl_ts, hardware_ts, data_type] + sample)
                    FAST_FINGER_COUNT += 1
                    FAST_SLOT_COUNTS[slot]["finger"] += 1

        elif 'IMU' in data_type:
            while len(sample) < 6:
                sample.append(0.)
            
            SLOT_SEEN[slot]["imu"] = True

            # Inversion Accel / Gyro car les données brutes sont inversées par rapport à l'en-tête
            sample = sample[3:6] + sample[0:3]

            key = f"imu_{slot}"
            with lsl_lock:
                o = lsl_outlets.get(key)
                if o: o.push_sample(sample, lsl_ts)

            if recording_local:
                buf = CSV_BUFFERS.get(key)
                if buf is not None:
                    buf.append([lsl_ts, hardware_ts, data_type] + sample)
                    FAST_IMU_COUNT += 1
                    FAST_SLOT_COUNTS[slot]["imu"] += 1

        sensor_queue.task_done()


def _slot_for(identifier, name):
    """Slot du registre pour un appareil qui se connecte, ou None si aucun slot ne porte son nom Bluetooth.
    À appeler sous state_lock. Ordre : même appareil qui se reconnecte, puis nom Bluetooth identique (casse ignorée)."""
    for slot, dev in state["devices"].items():
        if dev["id"] == identifier:
            return slot
    nm = (name or "").strip().lower()
    for d in DEVICE_REGISTRY:
        if d["name"] and d["name"].lower() == nm:
            return d["slot"]
    return None

def _warn_device(slot, msg):
    """Signale à l'examinateur une incohérence sur un slot (log + champ « warning » de /api/state)."""
    with state_lock:
        state["devices"][slot]["warning"] = msg
    log(f"⚠ {slot} : {msg}")

def _check_model(slot, identifier):
    """Vérifie, quelques secondes après la connexion, que les types de paquets reçus correspondent au modèle déclaré
    dans le registre. N'agit que si l'appareil est toujours celui qui a déclenché la vérification."""
    with state_lock:
        dev = state["devices"][slot]
        if not dev["connected"] or dev["id"] != identifier:
            return
    expected = SLOT_INFO[slot]["streams"]
    missing = [st for st in expected if not SLOT_SEEN[slot][st]]
    if missing:
        _warn_device(slot, f"aucun paquet « {', '.join(missing)} » reçu en {MODEL_CHECK_DELAY:.0f} s alors que le slot est déclaré "
                           f"{SLOT_INFO[slot]['model']} — vérifie le modèle dans le registre, ou si l'appareil est bien en mode raw")

def _reset_devices():
    """Marque tous les appareils comme déconnectés. À appeler sous state_lock."""
    for dev in state["devices"].values():
        dev["connected"] = False
        dev["id"]        = None
        dev["name"]      = None
        dev["warning"]   = None

def on_connect(identifier, name, fw):
    with state_lock:
        slot = _slot_for(identifier, name)
        if slot is None:
            log(f"Appareil ignoré : {name} — aucun slot ne porte ce nom Bluetooth "
                f"(ajoute-le dans l'examinateur, avec son type, avant de scanner)")
            return
        dev = state["devices"][slot]
        if dev["id"] is not None and dev["id"] != identifier:
            FAST_SLOT_MAP.pop(dev["id"], None)      # l'ancien appareil de ce slot est remplacé
        dev["connected"] = True
        dev["id"]        = identifier
        dev["name"]      = name
        dev["warning"]   = None
        SLOT_SEEN[slot]["finger"] = False
        SLOT_SEEN[slot]["imu"]    = False
        info  = SLOT_INFO[slot]
        label = f"{info['model']} {slot} ({info['position']})"

    _remember_tap(name, info["model"])
    FAST_SLOT_MAP[identifier] = slot
    
    # Petit délai pour solidifier la connexion Bluetooth avant d'envoyer la commande de mode brut
    threading.Timer(1.0, lambda: _set_mode(identifier, label, name)).start()
    threading.Timer(MODEL_CHECK_DELAY, lambda: _check_model(slot, identifier)).start()

def _set_mode(identifier, label, name):
    try:
        if tap:
            tap.set_input_mode(TapInputMode("raw", [0, 5, 4]), identifier)
            # log(f"✓ {label} ({name}) mode raw configuré")
    except Exception as e:
        log(f"⚠ Erreur set mode sur {label}: {e}")
    log(f"✓ {label} connecté : {name}")

def on_disconnect(identifier):
    with state_lock:
        for slot, dev in state["devices"].items():
            if dev["id"] == identifier:
                dev["connected"] = False
                dev["id"]        = None
                dev["name"]      = None
                dev["warning"]   = None
                log(f"⚠ {SLOT_INFO[slot]['model']} {slot} déconnecté")
                break
    FAST_SLOT_MAP.pop(identifier, None)

def on_raw_sensor_data(identifier, raw_sensor_data):
    """
    Fonction appelée par le thread Bluetooth.
    Prend les timestamps ultra rapidement, ne bloque rien, et pousse dans la queue.
    """
    # 1. Prendre les timestamps tout de suite (horloge logicielle ultra précise et horloge matérielle)
    lsl_ts      = local_clock()
    hardware_ts = getattr(raw_sensor_data, 'timestamp', 0) / 1000.0
    
    if not raw_sensor_data.points:
        return

    slot = FAST_SLOT_MAP.get(identifier)
    if not slot:
        return

    data_type = str(raw_sensor_data.type)
    
    # On met la donnée dans la queue pour la traiter de façon asynchrone
    sensor_queue.put((lsl_ts, hardware_ts, slot, data_type, raw_sensor_data.points, FAST_RECORDING))


def _init_tap():
    global tap, queue_thread_active
    
    # Démarrage du thread de traitement de la queue
    if not queue_thread_active:
        queue_thread_active = True
        threading.Thread(target=process_sensor_queue, daemon=True).start()
        
    tap = TapSDK()
    tap.register_connection_events(on_connect)
    tap.register_disconnection_events(on_disconnect)
    tap.register_raw_data_events(on_raw_sensor_data)

def _run_tap():
    global tap, tap_thread
    retry_count = 0
    while True:
        with state_lock:
            if not state["tap_scanning"]:
                break
        try:
            if tap is None:
                _init_tap()
            tap.run()
            # Maintient le thread actif pour ne pas déclencher le finally et détruire l'objet tap
            while True:
                with state_lock:
                    if not state["tap_scanning"]:
                        break
                time.sleep(1)
            break  # Sortie normale car tap_scanning est False
        except Exception as e:
            log(f"⚠ SDK erreur : {e}")
            retry_count += 1
            log(f"🔄 Tentative de reconnexion dans 3 secondes... ({retry_count})")
            time.sleep(3)
            # On force la réinitialisation au prochain tour
            if tap is not None:
                try: # Tente de nettoyer si possible
                    pass 
                except:
                    pass
            # Le tap sera recréé par relance de la boucle
    
    # Sortie 
    with state_lock:
        _reset_devices()
        state["tap_scanning"]   = False
        state["lsl_active"]     = False
    with lsl_lock:
        lsl_outlets.clear()
    tap = None
    tap_thread = None
    log("⚠ Bluetooth arrêté.")

# ═══════════════════════════════════════════════════════════
#  FLASK
# ═══════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template("index_customdata.html")

@app.route('/examiner')
def examiner():
    # Language choice: ?lang=fr|en (saved in a cookie), otherwise cookie, otherwise French
    lang = request.args.get('lang') or request.cookies.get('lang', 'fr')
    if lang not in ('fr', 'en'):
        lang = 'fr'
    resp = make_response(render_template('examiner_en.html' if lang == 'en' else 'examiner_customdata.html'))
    if request.args.get('lang') in ('fr', 'en'):
        resp.set_cookie('lang', lang, max_age=60*60*24*365)
    return resp

@app.route('/sets')
@app.route('/dataset')
def examiner_views():
    # Mêmes fichiers que /examiner : la page lit son URL et n'affiche que l'onglet voulu
    return render_template("examiner_customdata.html")

# ───── Versions anglaises (style B · Soft Modern) : /en, /en/examiner, /en/sets, /en/dataset, /en/classifier ─────
@app.route('/en')
def index_en():
    return render_template("index_customdata_en.html")

@app.route('/en/examiner')
@app.route('/en/sets')
@app.route('/en/dataset')
def examiner_en_views():
    # Même page pour les trois chemins, comme /examiner, /sets et /dataset
    return render_template("examiner_en.html")

@app.route('/en/classifier')
def classifier_en():
    for p in (os.path.join(BASE_DIR, "templates", "classifier_v2_en.html"),
              os.path.join(BASE_DIR, "classifier_v2_en.html")):
        if os.path.isfile(p):
            return send_file(p, mimetype="text/html")
    return "classifier_v2_en.html introuvable (le mettre dans templates/)", 404

@app.route('/api/state')
def get_state():
    with state_lock:
        order = state["gesture_order"]
        idx   = state["current_gesture_idx"]
        s = {
            "participant_id":      state["participant_id"],
            "phase":               state["phase"],
            "recording":           state["recording"],
            "current_gesture":     order[idx] if idx < len(order) else (order[-1] if order else None),
            "current_gesture_idx": idx,
            "current_repetition":  state["current_repetition"],
            "total_repetitions":   REPETITIONS_PER_GESTURE,
            "total_gestures":      len(order),
            "gesture_order":       order,
            "untested_configuration": UNTESTED_CONFIGURATION,
            "devices":             {sl: dict(dv, model=SLOT_INFO[sl]["model"], position=SLOT_INFO[sl]["position"], streams=list(SLOT_INFO[sl]["streams"]),
                                                pinned_name=SLOT_INFO[sl]["name"])
                                    for sl, dv in state["devices"].items()},
            "tap1_connected":      state["devices"].get("TAP1", {}).get("connected", False),   # anciens champs (v2)
            "tap2_connected":      state["devices"].get("TAP2", {}).get("connected", False),
            "tap_scanning":        state["tap_scanning"],
            "lsl_active":          state["lsl_active"],
            "finger_count":        FAST_FINGER_COUNT,
            "imu_count":           FAST_IMU_COUNT,
            "imu_counts":          {sl: c["imu"] for sl, c in FAST_SLOT_COUNTS.items()},
            "imu_count_tap1":      FAST_SLOT_COUNTS.get("TAP1", {}).get("imu", 0),              # anciens champs (v2)
            "imu_count_tap2":      FAST_SLOT_COUNTS.get("TAP2", {}).get("imu", 0),
            "countdown":           state["countdown"],
            "log":                 list(state["log"])[:10],
            "video_file":          state["video_file"],
            "history":             state["history"],
            "replay_requested":    state["replay_requested"],
            "session_number":      state["session_number"],
            "recording_elapsed":   round(time.time() - state["recording_start_time"], 1)
                                   if state["recording_start_time"] else 0,
            "gesture_set":         ACTIVE_SET,
            "dataset":             ACTIVE_DATASET,
            "gesture_labels":      {g["name"]: g.get("label", "") for g in GESTURE_DEFS},
            "current_gesture_def": GESTURE_MAP.get(order[idx]) if idx < len(order) else (GESTURE_MAP.get(order[-1]) if order else None),
        }
    return jsonify(s)

def _registry_edit_error():
    """Texte d'erreur si le registre ne peut pas être modifié maintenant, sinon None."""
    with state_lock:
        if state["tap_scanning"]:
            return "Arrête le scan (Reset) avant de modifier les appareils"
        if state["recording"] or state["countdown"] > 0:
            return "Impossible pendant l'enregistrement"
    return None

def _apply_registry(registry):
    """Remplace le registre et met à jour tout ce qui en dépend : slots, flux, buffers CSV, compteurs, état, devices.json."""
    DEVICE_REGISTRY[:] = registry
    _refresh_registry_views()
    with state_lock:
        old = state["devices"]
        state["devices"] = {s: old.get(s, {"connected": False, "id": None, "name": None, "warning": None}) for s in SLOTS}
    FAST_SLOT_COUNTS.clear()
    FAST_SLOT_COUNTS.update({s: {"finger": 0, "imu": 0} for s in SLOTS})
    SLOT_SEEN.clear()
    SLOT_SEEN.update({s: {"finger": False, "imu": False} for s in SLOTS})
    CSV_BUFFERS.clear()
    CSV_BUFFERS.update({k: [] for k in STREAM_KEYS})
    _save_registry()
    if UNTESTED_CONFIGURATION:
        log(f"⚠ {UNTESTED_NOTE}")

def _registry_entries():
    return [{k: d[k] for k in ("slot", "model", "position", "name")} for d in DEVICE_REGISTRY]

@app.route('/api/devices', methods=['GET'])
def devices_get():
    return jsonify({"devices": _registry_entries(), "models": list(DEVICE_MODELS),
                    "untested_configuration": UNTESTED_CONFIGURATION})

@app.route('/api/known_taps', methods=['GET'])
def known_taps_get():
    return jsonify({"taps": _known_taps()})

@app.route('/api/known_taps/forget', methods=['POST'])
def known_taps_forget():
    name = (request.get_json(silent=True) or {}).get("name")
    with _known_lock:
        known = [d for d in _known_taps() if d["name"] != name]
        with open(KNOWN_TAPS_FILE, "w", encoding="utf-8") as f:
            json.dump(known, f, ensure_ascii=False, indent=2)
    return jsonify({"ok": True})

@app.route('/api/devices/add', methods=['POST'])
def devices_add():
    err = _registry_edit_error()
    if err:
        return jsonify({"ok": False, "msg": err})
    body = request.get_json(silent=True) or {}
    # Le slot est créé vide : type par défaut TapStrap2, nom Bluetooth à saisir ensuite dans la carte du slot.
    try:
        _apply_registry(_normalize_registry(_registry_entries() + [
            {"slot": body.get("slot") or None, "model": body.get("model") or "TapStrap2",
             "position": body.get("position", ""), "name": body.get("name", "")}]))
    except ValueError as e:
        return jsonify({"ok": False, "msg": str(e)})
    log(f"Slot ajouté : {DEVICE_REGISTRY[-1]['slot']} ({DEVICE_REGISTRY[-1]['model']}) — choisis son type et son nom Bluetooth")
    return jsonify({"ok": True})

@app.route('/api/devices/remove', methods=['POST'])
def devices_remove():
    err = _registry_edit_error()
    if err:
        return jsonify({"ok": False, "msg": err})
    slot = (request.get_json(silent=True) or {}).get("slot")
    entries = [e for e in _registry_entries() if e["slot"] != slot]
    if len(entries) == len(DEVICE_REGISTRY):
        return jsonify({"ok": False, "msg": f"Slot inconnu : {slot}"})
    try:
        _apply_registry(_normalize_registry(entries))
    except ValueError as e:
        return jsonify({"ok": False, "msg": str(e)})
    log(f"Slot supprimé : {slot}")
    return jsonify({"ok": True})

@app.route('/api/devices/update', methods=['POST'])
def devices_update():
    err = _registry_edit_error()
    if err:
        return jsonify({"ok": False, "msg": err})
    body = request.get_json(silent=True) or {}
    entries, found = _registry_entries(), False
    for e in entries:
        if e["slot"] == body.get("slot"):
            found = True
            for k in ("name", "position", "model"):
                if k in body:
                    e[k] = body[k]
    if not found:
        return jsonify({"ok": False, "msg": f"Slot inconnu : {body.get('slot')}"})
    new_slot = str(body.get("new_slot") or "").strip()
    if new_slot and new_slot != body.get("slot"):
        # Changer de numéro : si le numéro visé est déjà pris, les deux slots échangent leur numéro.
        old = body.get("slot")
        for e in entries:
            if e["slot"] == new_slot:
                e["slot"] = old
            elif e["slot"] == old:
                e["slot"] = new_slot
        def _num(e):
            import re
            m = re.fullmatch(r"TAP(\d+)", e["slot"])
            return (0, int(m.group(1)), "") if m else (1, 0, e["slot"])
        entries.sort(key=_num)
    try:
        _apply_registry(_normalize_registry(entries))
    except ValueError as e:
        return jsonify({"ok": False, "msg": str(e)})
    log(f"Slot {body.get('slot')} mis à jour")
    return jsonify({"ok": True})

@app.route('/api/tap_connect', methods=['POST'])
def tap_connect():
    global tap, tap_thread
    with state_lock:
        if state["tap_scanning"]:
            return jsonify({"ok": False, "msg": "Déjà en écoute"})
        missing = _missing_names()
        if missing:
            return jsonify({"ok": False, "msg": f"Saisis le nom Bluetooth des slots : {', '.join(missing)}"})
        state["tap_scanning"] = True
    init_lsl_outlets()   # LSL en premier
    _init_tap()
    tap_thread = threading.Thread(target=_run_tap, daemon=True)
    tap_thread.start()
    if UNTESTED_CONFIGURATION:
        log(f"⚠ {UNTESTED_NOTE}")
    log("🔵 Scan BT démarré — allume les appareils Tap (" + ", ".join(f"{d['slot']}={d['model']}" for d in DEVICE_REGISTRY) + ")")
    return jsonify({"ok": True})

@app.route('/api/tap_disconnect', methods=['POST'])
def tap_disconnect():
    global tap, tap_thread
    with state_lock:
        _reset_devices()
        state["tap_scanning"]   = False
        state["lsl_active"]     = False
    with lsl_lock:
        lsl_outlets.clear()
    tap = None; tap_thread = None
    log("⚫ Connexions réinitialisées")
    return jsonify({"ok": True})

@app.route('/api/set_session', methods=['POST'])
def set_session():
    num = int(request.get_json().get('session_number', 1))
    with state_lock:
        state["session_number"] = num
    log(f"Session S{num:02d} définie")
    return jsonify({"ok": True})

@app.route('/api/set_repetition', methods=['POST'])
def set_repetition():
    rep = int(request.get_json().get('repetition', 1))
    with state_lock:
        if state["recording"] or state["countdown"] > 0:
            return jsonify({"ok": False, "msg": "Impossible pendant l'enregistrement"})
        # Les répétitions sont indexées à 0 dans l'état, mais l'utilisateur entre de 1 à REPETITIONS_PER_GESTURE
        state["current_repetition"] = max(0, min(rep - 1, REPETITIONS_PER_GESTURE - 1))
        # Si on était terminé, on repasse en "ready"
        if state["phase"] == "done":
            state["phase"] = "ready"
    log(f"Répétition fixée manuellement à la n°{rep}")
    return jsonify({"ok": True})

@app.route('/api/set_participant', methods=['POST'])
def set_participant():
    raw_pid = request.get_json().get('participant_id', '1')
    with state_lock:
        if str(raw_pid).lower() == 'test':
            pid = 'test'
            state["participant_id"]      = pid
            state["gesture_order"]       = LATIN_SQUARE[0]
        else:
            pid = int(raw_pid)
            state["participant_id"]      = pid
            state["gesture_order"]       = LATIN_SQUARE[(pid - 1) % len(LATIN_SQUARE)]
        
        state["current_gesture_idx"] = 0
        state["current_repetition"]  = 0
        state["phase"]               = "ready"
        state["history"]             = []
    if str(pid).lower() == 'test':
        log(f"Participant Test — ordre défini")
    else:
        log(f"Participant P{pid:02d} — ordre défini")
    return jsonify({"ok": True})

@app.route('/api/start', methods=['POST'])
def start_recording():
    if preproc["running"] or audit_job["running"]:
        return jsonify({"ok": False, "msg": "Prétraitement ou audit en cours — attendre la fin ou l'annuler"})
    if clf.job["running"]:
        return jsonify({"ok": False, "msg": "Un entraînement de classifieur est en cours — attendre la fin ou l'annuler (page Classifieur)"})
    with state_lock:
        if state["recording"] or state["countdown"] > 0:
            return jsonify({"ok": False, "msg": "Déjà en cours"})
        state["phase"]     = "countdown"
        state["countdown"] = COUNTDOWN_SECONDS

        gesture = state["gesture_order"][state["current_gesture_idx"]]
        rep     = state["current_repetition"]
        pid     = state["participant_id"]

    

    threading.Thread(target=_run_countdown, daemon=True).start()
    return jsonify({"ok": True})

def _run_countdown():
    global FAST_RECORDING, FAST_FINGER_COUNT, FAST_IMU_COUNT
    lsl_countdown_start = local_clock()
    with state_lock:
        state["lsl_countdown_start"] = lsl_countdown_start
        gesture = state["gesture_order"][state["current_gesture_idx"]]
        rep     = state["current_repetition"]
        pid     = state["participant_id"]
    prepare_csv_buffers()
    pid_str = "TEST" if str(pid).lower() == 'test' else f"P{int(pid):02d}"
    push_marker(f"COUNTDOWN_START|{pid_str}|{gesture}|rep{rep+1:02d}|s={COUNTDOWN_SECONDS}")

    for i in range(COUNTDOWN_SECONDS, 0, -1):
        with state_lock:
            state["countdown"] = i
        time.sleep(1)

    
    

    t_start = time.time()
    lsl_record_start = local_clock() 
    with state_lock:
        state["countdown"]              = 0
        state["recording"]              = True
        state["phase"]                  = "recording"
        state["recording_start_time"]   = t_start
        state["lsl_record_start"]       = lsl_record_start   # ← nouveau
        state["lsl_countdown_start"]    = local_clock() - COUNTDOWN_SECONDS  # ← quand start a été cliqué
        
    FAST_FINGER_COUNT = 0
    FAST_IMU_COUNT = 0
    for c in FAST_SLOT_COUNTS.values():
        c["finger"] = 0
        c["imu"]    = 0
    FAST_RECORDING = True
    
    
    push_marker(f"COUNTDOWN_END|{pid_str}|{gesture}|rep{rep+1:02d}", ts=lsl_record_start)
    push_marker(f"START|{pid_str}|{gesture}|rep{rep+1:02d}", ts=lsl_record_start)
    log(f"▶ START — {gesture} rep {rep+1}")
    
    # Enregistrer pendant 3 secondes
    with state_lock:
        gesture_idx = state["current_gesture_idx"]
        gesture_order = state["gesture_order"]
        gesture_name = gesture_order[gesture_idx]

    # Cherche la définition du geste pour sa durée
    gesture_def = next(
        (g for g in GESTURE_DEFS if g["name"] == gesture_name),
        None
    )
    duration = gesture_def.get("duration", RECORDING_DURATION) if gesture_def else RECORDING_DURATION

    log(f"Enregistrement {gesture_name} pendant {duration}s")
    time.sleep(duration)
    
    # Arrêter automatiquement
    _do_stop()

@app.route('/api/stop', methods=['POST'])
def stop_recording():
    return jsonify(_do_stop())

def _do_stop():
    global FAST_RECORDING
    FAST_RECORDING = False
    lsl_record_stop = local_clock()
    
    with state_lock:
        if not state["recording"]:
            return {"ok": False, "msg": "Pas en cours"}
        state["recording"] = False
        state["recording_start_time"] = None
        state["lsl_record_stop"] = lsl_record_stop
        fc      = FAST_FINGER_COUNT
        ic      = FAST_IMU_COUNT
        slot_counts = {sl: dict(c) for sl, c in FAST_SLOT_COUNTS.items()}
        gesture = state["gesture_order"][state["current_gesture_idx"]]
        rep     = state["current_repetition"]
        pid     = state["participant_id"]
        state["history"].append({
            "gesture": gesture, "rep": rep + 1,
            "finger_count": fc, "imu_count": ic,
            "slot_counts": slot_counts,
            "imu_count_tap1": slot_counts.get("TAP1", {}).get("imu", 0),   # anciens champs (v2)
            "imu_count_tap2": slot_counts.get("TAP2", {}).get("imu", 0),
            "ts": datetime.now().strftime('%H:%M:%S'),
            "status": "ok", "csv_paths": dict(state["last_csv_paths"]),
        })
        state["current_repetition"] += 1
        if state["current_repetition"] >= REPETITIONS_PER_GESTURE:
            state["current_repetition"]  = 0
            state["current_gesture_idx"] += 1
        is_last_step = state["current_gesture_idx"] >= len(state["gesture_order"])
        # On passe TOUJOURS par "ready" d'abord — y compris pour le tout dernier
        # geste — pour laisser au téléphone le temps de détecter la fin de
        # l'enregistrement (recording -> ready) et d'envoyer sa vidéo, exactement
        # comme pour tous les gestes précédents. Sans ça, le flag sautait direct
        # en "done" sur la dernière répétition et le téléphone ne voyait jamais
        # le signal "ready" qu'il attend pour déclencher l'upload de la vidéo.
        state["phase"] = "ready"
    write_and_close_csv_files()
    pid_str = "TEST" if str(pid).lower() == 'test' else f"P{int(pid):02d}"
    push_marker(f"STOP|{pid_str}|{gesture}|rep{rep+1:02d}|f={fc}|imu={ic}", ts=lsl_record_stop)
    log(f"■ STOP — {fc} pts doigts, {ic} pts IMU")
    if is_last_step:
        # Laisse 5s au téléphone pour finaliser/envoyer la dernière vidéo
        # avant de basculer l'UI en "done".
        threading.Timer(5.0, _finalize_done).start()
    return {"ok": True}

def _finalize_done():
    """Bascule en phase 'done' une fois la dernière vidéo censée être envoyée."""
    with state_lock:
        if state["phase"] == "ready" and state["current_gesture_idx"] >= len(state["gesture_order"]):
            state["phase"] = "done"
    log("✓ Expérience terminée (phase 'done')")

@app.route('/api/retry', methods=['POST'])
def retry_rep():
    with state_lock:
        if state["recording"]:
            return jsonify({"ok": False, "msg": "Stop d'abord"})
        if state["phase"] not in ["ready", "done"]:
            return jsonify({"ok": False, "msg": "Pas recommençable"})
        if state["history"]:
            state["history"][-1]["status"] = "retried"
        
        # If we are at repetition 0 of the current gesture but >0 gesture index,
        # moving back means we need to go to the PREVIOUS gesture at its LAST repetition
        if state["current_repetition"] == 0 and state["current_gesture_idx"] > 0:
             state["current_gesture_idx"] -= 1
             state["current_repetition"] = REPETITIONS_PER_GESTURE - 1
             state["phase"] = "ready" # If we were done, going back makes us ready
        elif state["current_repetition"] > 0:
            state["current_repetition"] -= 1
        elif state["current_repetition"] == 0 and state["current_gesture_idx"] == 0:
            pass # we are at the very beginning, cannot go back

        paths   = list(state["last_csv_paths"].values())
        
        # On ajoute le marker_log_path pour le déplacer lui aussi
        global marker_log_path
        if marker_log_path:
            paths.append(marker_log_path)
            
        video_path = state.get("last_video_path")
        gesture = state["gesture_order"][state["current_gesture_idx"]]
        rep     = state["current_repetition"]
        pid     = state["participant_id"]
    
    # Créer le dossier anomalie spécifique à ce participant (dans le dataset actif ; le participant « test » a le sien)
    if str(pid).lower() == 'test':
        anomalie_folder = os.path.join(BASE_DIR, "datatest", "anomalie", "test_P99")
    else:
        anomalie_folder = os.path.join(DATA_ROOT, "anomalie", f"P{int(pid):02d}")
    os.makedirs(anomalie_folder, exist_ok=True)
    import shutil
    
    # Déplacer les fichiers CSV refusés
    for p in paths:
        if p and os.path.exists(p):
            dest = os.path.join(anomalie_folder, os.path.basename(p))
            shutil.move(p, dest)
            state["history"][-1]["csv_paths"]["anomaly"] = dest

    # Déplacer la vidéo refusée
    if video_path and os.path.exists(video_path):
        dest_vid = os.path.join(anomalie_folder, os.path.basename(video_path))
        shutil.move(video_path, dest_vid)
        with state_lock:
            state["history"][-1]["video_path_anomaly"] = dest_vid

    pid_str = "TEST" if str(pid).lower() == 'test' else f"P{int(pid):02d}"
    push_marker(f"RETRY|{pid_str}|{gesture}|rep{rep+1:02d}")
    log(f"↩ RETRY — {gesture} rep {rep+1}")
    return jsonify({"ok": True})

@app.route('/api/skip', methods=['POST'])
def skip_gesture():
    with state_lock:
        if state["recording"]:
            return jsonify({"ok": False, "msg": "Stop d'abord"})
        gesture = state["gesture_order"][state["current_gesture_idx"]]
        pid     = state["participant_id"]
        state["history"].append({
            "gesture": gesture, "rep": state["current_repetition"] + 1,
            "finger_count": 0, "imu_count": 0,
            "ts": datetime.now().strftime('%H:%M:%S'),
            "status": "skipped", "csv_paths": {},
        })
        state["current_repetition"]  = 0
        state["current_gesture_idx"] += 1
        state["phase"] = ("done" if state["current_gesture_idx"] >= len(state["gesture_order"])
                          else "ready")
    pid_str = "TEST" if str(pid).lower() == 'test' else f"P{int(pid):02d}"
    push_marker(f"SKIP|{pid_str}|{gesture}")
    log(f"⏭ SKIP — {gesture}")
    return jsonify({"ok": True})

@app.route('/api/replay_video', methods=['POST'])
def replay_video():
    with state_lock:
        state["replay_requested"] = True
    log("↺ Replay vidéo")
    return jsonify({"ok": True})

@app.route('/api/replay_ack', methods=['POST'])
def replay_ack():
    with state_lock:
        state["replay_requested"] = False
    return jsonify({"ok": True})

def _is_inside(path, roots):
    """True si `path` (après résolution des liens et de `..`) est dans l'un des dossiers `roots`."""
    real = os.path.realpath(path)
    return any(real == r or real.startswith(r + os.sep) for r in (os.path.realpath(x) for x in roots))


@app.route('/api/plot_csv_by_path')
def plot_csv_by_path():
    path = request.args.get('path', '')
    allowed = [os.path.join(BASE_DIR, d) for d in ("data", "data_clean", "datatest")]
    if not path or not path.lower().endswith('.csv') or not _is_inside(path, allowed) or not os.path.isfile(path):
        return jsonify({"error": "Fichier introuvable ou non autorisé", "rows": []})
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= 2000: break
            rows.append(row)
    return jsonify({"file": os.path.basename(path), "rows": rows})

@app.route('/api/plot_csv')
def plot_csv():
    key  = f"{request.args.get('type','finger')}_{request.args.get('device','TAP1')}"
    with state_lock:
        path = state["last_csv_paths"].get(key)
    if not path or not os.path.exists(path):
        return jsonify({"error": f"Pas de fichier pour {key}", "rows": []})
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= 2000: break
            rows.append(row)
    return jsonify({"file": os.path.basename(path), "rows": rows})

@app.route('/api/set_video', methods=['POST'])
def set_video():
    with state_lock:
        state["video_file"] = request.get_json().get('filename', '')
    return jsonify({"ok": True})



@app.route('/api/cam_first_frame_ts', methods=['POST'])
def cam_first_frame_ts():
    # Reçoit les deux timestamps de la première frame :
    #   lsl_cam_start    : via System.nanoTime() → même horloge que le sync LSL NTP
    #   lsl_cam_start_hw : via timestamp hardware capteur camera2 → plus précis physiquement
    ts_sys = request.form.get('lsl_cam_start',    '0.0')
    ts_hw  = request.form.get('lsl_cam_start_hw', '0.0')
    try:
        ts_sys_f = float(str(ts_sys).strip().replace(',', '.'))
        ts_hw_f  = float(str(ts_hw ).strip().replace(',', '.'))
        with state_lock:
            state["lsl_cam_start_first_frame"]    = ts_sys_f
            state["lsl_cam_start_first_frame_hw"] = ts_hw_f
        hw_lag_ms = (ts_sys_f - ts_hw_f) * 1000
        log(f"📷 First frame sys={ts_sys_f:.6f}  hw={ts_hw_f:.6f}  lag={hw_lag_ms:.2f}ms")
        return jsonify({"ok": True})
    except Exception as e:
        log(f"⚠ Erreur parsing timestamp: {e}")
        return jsonify({"ok": False, "msg": f"Timestamp invalide: {ts_sys} / {ts_hw}"})



@app.route('/api/lsl_sync', methods=['GET'])
def lsl_sync():
    return jsonify({"lsl_now": local_clock()})



@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    video   = request.files.get('video')
    gesture = request.form.get('gesture', 'unknown')
    rep     = request.form.get('rep', '0')
    # t1_sys : System.nanoTime() → horloge de référence du sync LSL NTP Android↔Flask
    # t1_hw  : timestamp hardware capteur camera2 → moment physique exact de la 1re frame
    # t4     : System.nanoTime() au moment du stop
    lsl_cam_start    = request.form.get('lsl_cam_start',    '0.0')
    lsl_cam_start_hw = request.form.get('lsl_cam_start_hw', '0.0')
    lsl_cam_stop     = request.form.get('lsl_cam_stop',     '0.0')
 
    def _parse_timestamp(value):
        return float(str(value).strip().replace(',', '.'))
 
    if not video:
        return jsonify({"ok": False, "msg": "Pas de fichier"})
 
    with state_lock:
        ff_ts    = state.pop("lsl_cam_start_first_frame",    0.0)
        ff_ts_hw = state.pop("lsl_cam_start_first_frame_hw", 0.0)
        lsl_record_start = state.get("lsl_record_start", 0.0)
        lsl_record_stop  = state.get("lsl_record_stop",  0.0)
        pid              = int(state.get("participant_id", 1))
        session_num      = int(state.get("session_number", 1))
 
    with state_lock:
        video_pending_anomaly = state.pop("video_pending_anomaly", False)

    # ── t1_sys : préférer la valeur reçue en temps réel (cam_first_frame_ts)
    #             si disponible, sinon fallback sur la valeur dans l'upload
    t1_sys_val = ff_ts    if ff_ts    > 0 else _parse_timestamp(lsl_cam_start)
    t1_hw_val  = ff_ts_hw if ff_ts_hw > 0 else _parse_timestamp(lsl_cam_start_hw)
    t4_val     = _parse_timestamp(lsl_cam_stop)

    # Diagnostic du lag entre les deux méthodes de timestamping
    hw_lag_ms = (t1_sys_val - t1_hw_val) * 1000
    log(f"📊 t1_sys={t1_sys_val:.6f}  t1_hw={t1_hw_val:.6f}  lag={hw_lag_ms:.2f}ms")

    lsl_t1_sys = f"{t1_sys_val:.6f}".replace(".", "-")   # timestamp System.nanoTime()
    lsl_t1_hw  = f"{t1_hw_val:.6f}".replace(".", "-")    # timestamp hardware capteur
    lsl_t2     = f"{lsl_record_start:.6f}".replace(".", "-")
    lsl_t3     = f"{lsl_record_stop:.6f}".replace(".", "-")
    lsl_t4     = f"{t4_val:.6f}".replace(".", "-")
 
    if video_pending_anomaly:
        folder      = os.path.join(DATA_ROOT, "anomalie", f"P{pid:02d}")
        anomaly_str = "_RETRY"
    else:
        folder      = os.path.join(DATA_ROOT, f"P{pid:02d}", "videos")
        anomaly_str = ""
        
    os.makedirs(folder, exist_ok=True)
 
    filename = (
        f"P{pid:02d}_S{session_num:02d}"
        f"_{gesture.replace(' ', '_')}"
        f"_rep{int(rep):02d}"
        f"_t1sys{lsl_t1_sys}"     # System.nanoTime() → LSL — horloge sync NTP
        f"_t1hw{lsl_t1_hw}"       # hardware capteur → LSL — moment physique exact
        f"_t2flaskStart{lsl_t2}"
        f"_t3flaskStop{lsl_t3}"
        f"_t4camStop{lsl_t4}"
        f"{anomaly_str}"
        f".mp4"
    )
 
    main_path = os.path.join(folder, filename)
    video.save(main_path)
    
    with state_lock:
        state["last_video_path"] = main_path
 
    log(f"Vidéo reçue: {filename}")
    return jsonify({"ok": True, "filename": filename})



@app.route('/clock')
def lsl_clock():
    return r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>LSL Clock</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
 
  body {
    background: #0a0a0a;
    color: #e8e8e8;
    font-family: 'Courier New', monospace;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    gap: 32px;
  }
 
  h1 {
    font-size: 13px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #444;
    font-weight: normal;
  }
 
  /* ── Horloge principale ── */
  #clock {
    font-size: clamp(40px, 8vw, 80px);
    font-weight: bold;
    letter-spacing: 2px;
    color: #4fffb0;
    text-shadow: 0 0 30px #4fffb044;
    transition: color 0.1s;
  }
 
  #clock.flash { color: #fff; text-shadow: 0 0 40px #4fffb0aa; }
 
  /* ── Infos secondaires ── */
  .info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px 32px;
    font-size: 13px;
    color: #555;
  }
  .info-grid span { color: #888; }
 
  /* ── Marqueurs ── */
  #markers {
    width: min(600px, 90vw);
    max-height: 220px;
    overflow-y: auto;
    border: 1px solid #1e1e1e;
    border-radius: 6px;
    padding: 10px 14px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
 
  .marker-row {
    display: flex;
    gap: 14px;
    font-size: 12px;
    animation: fadein 0.3s ease;
  }
  @keyframes fadein { from { opacity:0; transform:translateY(-4px); } to { opacity:1; } }
 
  .marker-ts  { color: #4fffb0; min-width: 120px; flex-shrink: 0; }
  .marker-val { color: #aaa; word-break: break-all; }
 
  #markers-label {
    font-size: 11px;
    letter-spacing: 3px;
    color: #333;
    text-transform: uppercase;
    align-self: flex-start;
  }
 
  /* ── Bouton ── */
  button {
    background: none;
    border: 1px solid #333;
    color: #555;
    font-family: inherit;
    font-size: 11px;
    letter-spacing: 2px;
    padding: 8px 20px;
    cursor: pointer;
    text-transform: uppercase;
    transition: all 0.2s;
  }
  button:hover { border-color: #4fffb0; color: #4fffb0; }
</style>
</head>
<body>
 
<h1>LSL local_clock — live</h1>
 
<div id="clock">—</div>
 
<div class="info-grid">
  <div>offset serveur</div> <span id="offset">—</span>
  <div>RTT HTTP</div>       <span id="rtt">—</span>
  <div>drift/s</div>        <span id="drift">—</span>
  <div>syncs</div>          <span id="syncs">0</span>
</div>
 
<div id="markers-label">derniers markers LSL</div>
<div id="markers"><span style="color:#333;font-size:12px">en attente…</span></div>
 
<button onclick="forceSync()">↺ resync</button>
 
<script>
// ── État ──────────────────────────────────────────────────────
let offset      = null   // local_clock_flask - performance.now()/1000
let lastOffset  = null
let syncCount   = 0
let knownMarkers = []
 
// ── Sync horloge (NTP simplifié) ─────────────────────────────
async function sync() {
  try {
    const t0   = performance.now()
    const resp = await fetch('/api/lsl_sync')
    const t1   = performance.now()
    const data = await resp.json()
 
    const rtt_s  = (t1 - t0) / 1000
    const t_mid  = t0 / 1000 + rtt_s / 2
    const newOff = data.lsl_now - t_mid
 
    lastOffset = offset
    offset     = newOff
    syncCount++
 
    document.getElementById('rtt').textContent    = (rtt_s * 1000).toFixed(1) + ' ms'
    document.getElementById('offset').textContent = newOff.toFixed(6) + ' s'
    document.getElementById('syncs').textContent  = syncCount
 
    if (lastOffset !== null) {
      const drift = (newOff - lastOffset) / 10   // drift par seconde (sync toutes les 10s)
      document.getElementById('drift').textContent =
        (drift * 1e6).toFixed(1) + ' µs/s'
    }
  } catch(e) {
    document.getElementById('offset').textContent = 'erreur réseau'
  }
}
 
function forceSync() { sync() }
 
// Sync initiale + périodique toutes les 10s
sync()
setInterval(sync, 10_000)
 
// ── Affichage horloge (RAF = ~60fps) ─────────────────────────
const clockEl = document.getElementById('clock')
 
function tick() {
  if (offset !== null) {
    const lsl = performance.now() / 1000 + offset
    clockEl.textContent = lsl.toFixed(6)
  }
  requestAnimationFrame(tick)
}
requestAnimationFrame(tick)
 
// ── Poll markers toutes les 500ms ────────────────────────────
async function pollMarkers() {
  try {
    const resp = await fetch('/api/last_markers?n=10')
    const data = await resp.json()
    const markers = data.markers || []
 
    // Affiche seulement les nouveaux
    const newOnes = markers.filter(m => !knownMarkers.includes(m.ts + m.val))
    if (newOnes.length === 0) return
 
    newOnes.forEach(m => knownMarkers.push(m.ts + m.val))
 
    const container = document.getElementById('markers')
    if (container.querySelector('span')) container.innerHTML = ''
 
    newOnes.forEach(m => {
      const row = document.createElement('div')
      row.className = 'marker-row'
      row.innerHTML = `<span class="marker-ts">${parseFloat(m.ts).toFixed(6)}</span>
                       <span class="marker-val">${m.val}</span>`
      container.prepend(row)
    })
 
    // Flash de l'horloge à chaque nouveau marker
    clockEl.classList.add('flash')
    setTimeout(() => clockEl.classList.remove('flash'), 150)
 
  } catch(e) {}
}
setInterval(pollMarkers, 500)
</script>
</body>
</html>
"""
 
 
@app.route('/api/last_markers')
def last_markers():
    """Retourne les N derniers markers depuis le fichier log actif."""
    n = int(request.args.get('n', 10))
    with marker_log_lock:
        path = marker_log_path
    if not path or not os.path.exists(path):
        # Pas de session active : cherche le marker log le plus récent dans le dataset actif
        files = sorted(glob.glob(os.path.join(DATA_ROOT, '**', '*_markers.csv'), recursive=True), key=os.path.getmtime)
        if not files:
            return jsonify({"markers": []})
        path = files[-1]
    rows = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    rows.append({"ts": row[0], "val": row[1]})
        return jsonify({"markers": rows[-n:]})
    except Exception as e:
        return jsonify({"markers": [], "error": str(e)})
 

# ═══════════════════════════════════════════════════════════
#  PRÉTRAITEMENT  (data/PXX → data_clean/Gesture_XX/PXX)
#  Tâche de fond : refusée pendant un essai, et l'inverse aussi
#  (le thread Bluetooth ne doit pas être en concurrence avec le CPU
#  du rééchantillonnage — même principe que l'architecture de capture).
# ═══════════════════════════════════════════════════════════
# VDATA / CLEAN_ROOT : dataset AFFICHÉ (voir _set_view), par défaut le dataset actif

preproc = {
    "running": False, "kind": None, "cancel": threading.Event(),
    "done": 0, "total": 0, "current": "", "ok": 0, "skipped": 0, "errors": 0, "warnings": 0,
    "started": None, "finished": None, "message": "", "recent_errors": [], "summary": None, "notices": [],
    
}
preproc_lock = threading.Lock()


def _preproc_progress(p):
    last = p["last"]
    with preproc_lock:
        preproc["done"], preproc["total"], preproc["current"] = p["done"], p["total"], p["current"]
        if last["status"] == "ok":
            preproc["ok"] += 1
            preproc["warnings"] += len(last["warnings"])
        elif last["status"] == "skipped":
            preproc["skipped"] += 1
        else:
            preproc["errors"] += 1
            preproc["recent_errors"] = (preproc["recent_errors"] + [f'{last["key"]} — {last["error"]}'])[-8:]


def _preproc_worker(kind, params):
    try:
        out_root = params["out_root"]
        if kind == "clean":
            diff = pp.stream_differences(pp.stream_sets(VDATA, params["participants"]))
            if diff:                              # simple notification : les répétitions sont nettoyées quand même
                note = pp.format_stream_notice(diff)
                with preproc_lock:
                    preproc["notices"] = [note]
                log(f"⚠ {len(diff['odd'])} répétition(s) avec des flux différents (voir la page Dataset)")
            results = pp.run_batch(VDATA, out_root, params["participants"], params["freq_hz"],
                                   params["method"], params["overwrite"], params["keep_intermediate"],
                                   fix_slope=bool(params.get("fix_slope")),
                                   progress=_preproc_progress, cancel=preproc["cancel"],
                                   workers=params.get("workers"))
            summary = pp.write_report(results, out_root)
            msg = f'Terminé : {preproc["ok"]} traitée(s), {preproc["skipped"]} ignorée(s), {preproc["errors"]} erreur(s)'
        else:
            summary, msg = None, ""
        with preproc_lock:
            preproc["summary"], preproc["message"] = summary, msg
        log(msg)
    except pp.Cancelled as e:
        try:                                      # le travail déjà fait est consigné : rapport cohérent avec les fichiers présents
            if getattr(e, "results", None):
                pp.write_report(e.results, params["out_root"])
        except Exception as ex_:
            log(f"Rapport partiel non écrit : {ex_}")
        with preproc_lock:
            preproc["message"] = "Annulé"
        log("Prétraitement annulé")
    except Exception as e:
        with preproc_lock:
            preproc["message"] = f"Erreur : {e}"
            preproc["errors"] += 1
        log(f"Prétraitement : erreur {e}")
    finally:
        with preproc_lock:
            preproc["running"] = False
            preproc["finished"] = time.time()


def _start_preproc(kind, params):
    with state_lock:
        busy = state["recording"] or state["countdown"] > 0
    if audit_job["running"]:
        return jsonify({"ok": False, "msg": "Un audit d'intégrité est en cours"})
    if clf.job["running"]:
        return jsonify({"ok": False, "msg": "Un entraînement de classifieur est en cours"})
    if busy:
        return jsonify({"ok": False, "msg": "Un essai est en cours — impossible de prétraiter maintenant"})
    with preproc_lock:
        if preproc["running"]:
            return jsonify({"ok": False, "msg": "Un prétraitement est déjà en cours"})
        preproc.update(running=True, kind=kind, done=0, total=0, current="", ok=0, skipped=0,
                       errors=0, warnings=0, started=time.time(), finished=None, message="",
                       recent_errors=[], summary=None, notices=[])
        preproc["cancel"].clear()
    if kind == "clean":
        _select_clean_version(params["version_slug"])        # l'app suit la version qu'on est en train de produire
    threading.Thread(target=_preproc_worker, args=(kind, params), daemon=True).start()
    return jsonify({"ok": True})


# ── Versions nommées du dataset nettoyé ──────────────────────────────────────
# Un même dataset brut peut avoir plusieurs versions propres (fréquence, interpolation…), chacune dans
# <data_clean/jeu>/<nom>/ (dossier de premier niveau, version.json y indique son dataset) avec ses propres statistiques et modèles. Sans nom, on écrit
# dans la racine du dataset nettoyé (comportement d'origine).
def _version_root(slug):
    return dt.version_root(CLEAN_BASE, slug)

def _select_clean_version(slug):
    global CLEAN_VERSION, CLEAN_ROOT
    CLEAN_VERSION, CLEAN_ROOT = slug, _version_root(slug)
    try:
        clf.reset_selected()              # le classifieur suit la version active
    except Exception:
        pass

def _count_clean(root):
    n = 0
    for g in glob.glob(os.path.join(root, "Gesture_*")):
        n += len(glob.glob(os.path.join(g, "P*", "*_Clean.csv")))
    return n

def _list_clean_versions():
    out = [{"slug": "", "name": "Par défaut", "files": _count_clean(CLEAN_BASE)}]
    if True:
        for n, root in dt.list_versions(CLEAN_BASE):
            try:
                with open(os.path.join(root, "version.json"), encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
            out.append({"slug": n, "name": meta.get("name") or n, "files": _count_clean(root),
                        "freq_hz": meta.get("freq_hz"), "method": meta.get("method"), "clock_corrected": bool(meta.get("clock_corrected")), "fix_slope": bool(meta.get("fix_slope")), "updated": meta.get("updated")})
    return out

@app.route('/api/preprocess/versions')
def preprocess_versions():
    return jsonify({"active": CLEAN_VERSION, "versions": _list_clean_versions()})

@app.route('/api/preprocess/version/select', methods=['POST'])
def preprocess_version_select():
    slug = str((request.get_json(silent=True) or {}).get("slug") or "")
    with state_lock:
        busy = state["recording"] or state["countdown"] > 0
    if busy or preproc["running"] or audit_job["running"] or clf.job["running"]:
        return jsonify({"ok": False, "msg": "Un essai, un prétraitement, un audit ou un entraînement est en cours"})
    if slug and slug not in dict(dt.list_versions(CLEAN_BASE)):
        return jsonify({"ok": False, "msg": "Version inconnue"})
    _select_clean_version(slug)
    log("Dataset nettoyé actif : " + (slug or "par défaut"))
    return jsonify({"ok": True})

@app.route('/api/preprocess/participants')
def preprocess_participants():
    parts = pp.list_participants(VDATA)
    vname = (request.args.get("version") or "").strip()      # nom en cours de saisie (pas encore créé) → compteurs de cette version
    root = _version_root(gs.slug(vname)) if vname else CLEAN_ROOT
    return jsonify({"participants": pp.count_pending(VDATA, root, parts),
                    "defaults": {"freq_hz": pp.DEFAULT_FREQ_HZ, "method": pp.DEFAULT_METHOD,
                                 "workers": pp.default_workers(), "cpus": os.cpu_count() or 1,
                                 "methods": list(pp.METHODS)}})


@app.route('/api/preprocess/start', methods=['POST'])
def preprocess_start():
    d = request.get_json(silent=True) or {}
    available = pp.list_participants(VDATA)
    wanted = d.get("participants") or available
    parts = [p for p in wanted if p in available]
    if not parts:
        return jsonify({"ok": False, "msg": "Aucun participant à traiter"})
    method = d.get("method", pp.DEFAULT_METHOD)
    if method not in pp.METHODS:
        return jsonify({"ok": False, "msg": f"Méthode inconnue : {method}"})
    try:
        freq = int(d.get("freq_hz", pp.DEFAULT_FREQ_HZ))
        assert 20 <= freq <= 1000
    except Exception:
        return jsonify({"ok": False, "msg": "Fréquence invalide (20–1000 Hz)"})
    vname = str(d.get("version") or "").strip()[:60]
    vslug = gs.slug(vname) if vname else CLEAN_VERSION          # nom vide : la version active
    out_root = _version_root(vslug)
    if vslug and vslug in SETS.datasets(os.path.basename(os.path.dirname(CLEAN_BASE))):
        return jsonify({"ok": False, "msg": f"« {vslug} » est déjà le nom d'un dataset : choisis un autre nom de version"})
    if vslug and dt.owner_of(out_root) not in (None, os.path.normpath(CLEAN_BASE)) \
            and os.path.normcase(dt.owner_of(out_root)) != os.path.normcase(os.path.normpath(CLEAN_BASE)):
        return jsonify({"ok": False, "msg": f"« {vslug} » est déjà une version d'un autre dataset : choisis un autre nom"})
    if vslug and os.path.isdir(out_root) and not os.path.isfile(os.path.join(out_root, "version.json")) and os.listdir(out_root):
        return jsonify({"ok": False, "msg": f"Le dossier {out_root} existe déjà et n'est pas une version : choisis un autre nom"})
    fix_slope = bool(d.get("fix_slope"))
    if fix_slope and not vslug:
        return jsonify({"ok": False, "msg": "Pente a = 1 fixe : donne un nom de version (le dataset par défaut garde la pente ajustée)"})
    meta = {}
    if vslug:
        try:
            with open(os.path.join(out_root, "version.json"), encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
        if _count_clean(out_root) > 0 and not meta.get("clock_corrected") and not d.get("overwrite"):
            # fichiers produits avant que l'horloge corrigée devienne le seul mode (temps d'arrivée) : ne pas les mélanger
            return jsonify({"ok": False, "msg": "Cette version contient des fichiers produits avec l'ancien temps d'arrivée (sans horloge corrigée) : "
                                                "choisis un autre nom, ou coche « Retraiter les répétitions déjà nettoyées »"})
        if _count_clean(out_root) > 0 and bool(meta.get("fix_slope")) != fix_slope and not d.get("overwrite"):
            return jsonify({"ok": False, "msg": "Cette version contient déjà des fichiers avec l'autre réglage de pente (ajustée / a = 1) : "
                                                "choisis un autre nom, ou coche « Retraiter les répétitions déjà nettoyées »"})
    if vslug:
        os.makedirs(out_root, exist_ok=True)
        mpath = os.path.join(out_root, "version.json")
        if not meta:
            meta = {"created": datetime.now().isoformat(timespec="seconds")}
        meta.update(dataset=os.path.basename(os.path.normpath(CLEAN_BASE)), name=vname or meta.get("name") or vslug, freq_hz=freq, method=method, clock_corrected=True, fix_slope=fix_slope,
                    updated=datetime.now().isoformat(timespec="seconds"))
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return _start_preproc("clean", {"participants": parts, "freq_hz": freq, "method": method,
                                    "version_slug": vslug, "out_root": out_root,
                                    "overwrite": bool(d.get("overwrite")),
                                    "keep_intermediate": bool(d.get("keep_intermediate")),
                                    "fix_slope": fix_slope,
                                    "workers": max(1, min(int(d.get("workers") or pp.default_workers()), 16))})


@app.route('/api/preprocess/cancel', methods=['POST'])
def preprocess_cancel():
    preproc["cancel"].set()
    return jsonify({"ok": True})


@app.route('/api/preprocess/status')
def preprocess_status():
    with preproc_lock:
        s = {k: v for k, v in preproc.items() if k != "cancel"}
    s["elapsed"] = round((s["finished"] or time.time()) - s["started"], 1) if s["started"] else 0
    if s["summary"] is None:
        path = os.path.join(CLEAN_ROOT, "preprocessing_summary.json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    s["summary"] = json.load(f)
            except Exception:
                pass
    return jsonify(s)


@app.route('/api/preprocess/report.csv')
def preprocess_report():
    path = os.path.join(CLEAN_ROOT, "preprocessing_report.csv")
    if not os.path.isfile(path):
        return jsonify({"ok": False, "msg": "Pas encore de rapport"}), 404
    return send_file(path, mimetype="text/csv", as_attachment=True, download_name="preprocessing_report.csv")



# ═══════════════════════════════════════════════════════════
#  DATASET : inventaire, contrôle 5 répétitions, déplacement, aperçu
# ═══════════════════════════════════════════════════════════
audit_job = {"running": False, "cancel": threading.Event(), "done": 0, "total": 0, "current": "",
             "message": "", "started": None, "finished": None}
audit_lock = threading.Lock()


def _dataset_locked():
    with state_lock:
        rec = state["recording"] or state["countdown"] > 0
    if rec:
        return "Un essai est en cours — réessayer après l'enregistrement"
    if preproc["running"]:
        return "Un prétraitement est en cours"
    if audit_job["running"]:
        return "Un audit d'intégrité est en cours"
    if clf.job["running"]:
        return "Un entraînement de classifieur est en cours"
    return None


def _audit_worker():
    try:
        def prog(p):
            with audit_lock:
                audit_job.update(done=p["done"], total=p["total"], current=p["current"])
        s = dt.audit_all(VDATA, CLEAN_ROOT, progress=prog, cancel=audit_job["cancel"])
        msg = f'Audit terminé : {s["audited"]} répétitions, {s["with_errors"]} avec erreur(s), {s["with_warnings_only"]} avec avertissement(s)'
    except dt.AuditCancelled:
        msg = "Audit annulé"
    except Exception as e:
        msg = f"Erreur d'audit : {e}"
    with audit_lock:
        audit_job.update(running=False, message=msg, finished=time.time())
    log(msg)


@app.route('/api/dataset/audit/start', methods=['POST'])
def audit_start():
    lock = _dataset_locked()
    if lock:
        return jsonify({"ok": False, "msg": lock})
    with audit_lock:
        audit_job.update(running=True, done=0, total=0, current="", message="", started=time.time(), finished=None)
        audit_job["cancel"].clear()
    threading.Thread(target=_audit_worker, daemon=True).start()
    return jsonify({"ok": True})


@app.route('/api/dataset/audit/cancel', methods=['POST'])
def audit_cancel():
    audit_job["cancel"].set()
    return jsonify({"ok": True})


@app.route('/api/dataset/audit/status')
def audit_status():
    with audit_lock:
        j = {k: v for k, v in audit_job.items() if k != "cancel"}
    j["summary"] = dt.load_audit_summary(CLEAN_ROOT)
    return jsonify(j)


@app.route('/api/dataset/audit/issues')
def audit_issues_route():
    limit = min(int(request.args.get('limit', 300) or 300), 5000)
    return jsonify({"items": dt.audit_issues(CLEAN_ROOT, request.args.get('cls', ''), request.args.get('code', ''),
                                             limit, request.args.get('severity', ''))})


@app.route('/api/dataset/audit/report.csv')
def audit_report():
    path = os.path.join(CLEAN_BASE, "integrity_issues.csv")
    if not os.path.isfile(path):
        return jsonify({"ok": False, "msg": "Pas encore d'audit"}), 404
    return send_file(path, mimetype="text/csv", as_attachment=True, download_name="integrity_issues.csv")


@app.route('/api/dataset/overview')
def dataset_overview():
    return jsonify(dt.scan_overview(VDATA, CLEAN_ROOT, VGESTURES, VREPS))


@app.route('/api/dataset/cell')
def dataset_cell():
    try:
        return jsonify(dt.cell_detail(VDATA, CLEAN_ROOT, request.args.get('participant', ''),
                                      request.args.get('gesture', ''), VREPS))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/dataset/move', methods=['POST'])
def dataset_move():
    lock = _dataset_locked()
    if lock:
        return jsonify({"ok": False, "msg": lock})
    d = request.get_json(silent=True) or {}
    try:
        res = dt.move_repetitions(VDATA, CLEAN_ROOT, d.get("participant", ""),
                                  d.get("keys") or [], str(d.get("reason", ""))[:200])
    except ValueError as e:
        return jsonify({"ok": False, "msg": str(e)})
    log(f"Dataset : {len(res['moved'])} répétition(s) déplacée(s) vers anomalie ({d.get('participant')})")
    return jsonify({"ok": True, **res})


@app.route('/api/dataset/anomalies')
def dataset_anomalies():
    return jsonify({"items": dt.list_anomalies(VDATA, request.args.get('participant') or None)})


@app.route('/api/dataset/restore', methods=['POST'])
def dataset_restore():
    lock = _dataset_locked()
    if lock:
        return jsonify({"ok": False, "msg": lock})
    d = request.get_json(silent=True) or {}
    try:
        return jsonify({"ok": True, **dt.restore_repetition(VDATA, d.get("participant", ""), d.get("key", ""))})
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route('/api/dataset/stats')
def dataset_stats():
    return jsonify(dt.stats_rows(CLEAN_ROOT))


@app.route('/api/dataset/rep')
def dataset_rep():
    try:
        return jsonify(dt.rep_data(VDATA, CLEAN_ROOT, request.args.get('participant', ''),
                                   request.args.get('key', ''), request.args.get('source', 'raw')))
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 404


# ═══════════════════════════════════════════════════════════
#  CLASSIFIEUR  (page /classifier + /api/clf/*)
#  Même règle que le prétraitement : jamais en même temps qu'un essai.
# ═══════════════════════════════════════════════════════════
def _clf_busy():
    with state_lock:
        rec = state["recording"] or state["countdown"] > 0
    if rec:
        return "Un essai est en cours — impossible d'entraîner ou d'inférer maintenant"
    if preproc["running"]:
        return "Un prétraitement est en cours"
    if audit_job["running"]:
        return "Un audit d'intégrité est en cours"
    return None


clf.register(app, lambda: CLEAN_ROOT, BASE_DIR, busy=_clf_busy)


# ── Dataset affiché (indépendant de celui de l'enregistrement) ──────────────
@app.route('/api/view')
def view_get():
    sets = []
    for s_ in SETS.summary(ACTIVE_SET, ACTIVE_DATASET):
        if s_.get("error"):
            continue
        sets.append({"id": s_["id"], "name": s_["name"],
                     "datasets": [{"id": d["id"], "name": d["name"], "active": bool(d.get("active"))} for d in s_["datasets"]]})
    return jsonify({"sets": sets, "active": {"set": ACTIVE_SET, "dataset": ACTIVE_DATASET},
                    "view": {"set": VIEW_SET or ACTIVE_SET, "dataset": VIEW_DS or ACTIVE_DATASET, "following": VIEW_SET is None}})

@app.route('/api/view/select', methods=['POST'])
def view_select():
    d = request.get_json(silent=True) or {}
    if preproc["running"] or audit_job["running"] or clf.job["running"]:
        return jsonify({"ok": False, "msg": "Un prétraitement, un audit ou un entraînement est en cours : attends la fin ou annule"})
    try:
        _set_view(d.get("set") or None, d.get("dataset") or None)
    except ValueError as e:
        return jsonify({"ok": False, "msg": str(e)})
    try:
        clf.reset_selected()
    except Exception:
        pass
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════
#  JEUX DE GESTES  (onglet « Jeux de gestes » de l'examinateur)
# ═══════════════════════════════════════════════════════════
def _sets_busy():
    with state_lock:
        if state["recording"] or state["countdown"] > 0:
            return "Un essai est en cours — impossible de changer de jeu de gestes maintenant"
    if clf.job.get("running"):
        return "Un entraînement est en cours"
    return _clf_busy()

def _on_set_activated(sid, ds=None):
    _apply_set(sid, ds)
    try:
        clf.reset_selected()              # le classifieur suit de nouveau le dataset actif
    except Exception:
        pass
    with state_lock:                      # le participant en cours n'a plus de sens dans un autre jeu / dataset
        state["participant_id"]      = 1
        state["gesture_order"]       = LATIN_SQUARE[0]
        state["current_gesture_idx"] = 0
        state["current_repetition"]  = 0
        state["phase"]               = "idle"
        state["history"]             = []
    log(f"Jeu de gestes actif : {SETS.get(sid, ds)['name']} · dataset {ACTIVE_DATASET}")

gs.register(app, SETS, _sets_busy, lambda: ACTIVE_SET, _on_set_activated, os.path.join(BASE_DIR, "static", "videos"),
            lambda: ACTIVE_DATASET)


if __name__ == '__main__':
    print("=" * 52)
    print("  TapStrap Experiment — Flask + LSL")
    print("  Participant : http://localhost:5000")
    print("  Examinateur : http://localhost:5000/examiner")
    print("  Classifieur : http://localhost:5000/classifier")
    print("  English     : http://localhost:5000/en  (/en/examiner, /en/classifier)")
    print("=" * 52)
    print("\n  LSL streams disponibles dans LabRecorder dès")
    print("  que tu cliques 'Connecter' côté examinateur.\n")
    print("  Streams : " + ", ".join(f"TapStrap_{'Finger' if k.startswith('finger_') else 'IMU'}_{k.split('_', 1)[1]}" for k in STREAM_KEYS))
    print("            TapExp_Markers\n")
    print("  Prétraitement : onglet « Dataset » de l'examinateur\n")
    
   
    app.run(debug=False,host='0.0.0.0', port=5000, threaded=True)