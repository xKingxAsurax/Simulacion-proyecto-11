import os
import sys
from tensorflow import keras
# Configurar UTF-8 en salida de consola Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
import json
from pathlib import Path

# Fijar semilla para reproducibilidad
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

IMG_HEIGHT = 160
IMG_WIDTH = 160
BATCH_SIZE = 32
EPOCHS = 10
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"

print("=" * 70)
print(" 1. CARGA DE DATASET DIVIDIDO (TRAIN 70%, VAL 15%, TEST NO VISTOS 15%)")
print("=" * 70)

train_dir = os.path.join(DATASET_DIR, "train")
val_dir = os.path.join(DATASET_DIR, "val")
test_dir = os.path.join(DATASET_DIR, "test")

missing_dirs = [
    directory for directory in (train_dir, val_dir, test_dir)
    if not os.path.isdir(directory)
]
if missing_dirs:
    missing_text = "\n".join(f"  - {directory}" for directory in missing_dirs)
    raise FileNotFoundError(
        "No se encontró el dataset dividido. Ejecuta primero "
        f"{PROJECT_DIR / 'prepare_dataset.py'} o verifica estas carpetas:\n"
        f"{missing_text}"
    )

train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# IMPORTANTE: test_ds DEBE TENER shuffle=False para que las predicciones
# coincidan exactamente con las etiquetas reales de los archivos no vistos
test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\n[+] Clases detectadas ({num_classes}): {class_names}")

# Guardar classes.json y classes.txt
with open(PROJECT_DIR / "classes.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)
with open(PROJECT_DIR / "classes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(class_names))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("\n" + "=" * 70)
print(" 2. DISEÑO DE LA RED NEURONAL CONVOLUCIONAL (CNN - SIN MODELOS PRE-ENTRENADOS)")
print("=" * 70)

# Data augmentation integrada para robustez visual
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.03),
    layers.RandomTranslation(0.02, 0.02),
], name="data_augmentation")

model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="input_documento"),
    data_augmentation,
    layers.Rescaling(1.0 / 255.0, name="normalizacion"),

    # Bloque 1: Bordes y encabezados
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1"),
    layers.MaxPooling2D((2, 2), name="pool1"),

    # Bloque 2: Tablas y bloques de texto
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2"),
    layers.MaxPooling2D((2, 2), name="pool2"),

    # Bloque 3: Distribución estructural de documentos
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="conv3"),
    layers.MaxPooling2D((2, 2), name="pool3"),

    # Bloque 4: Características visuales de alto nivel
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="conv4"),
    layers.MaxPooling2D((2, 2), name="pool4"),

    layers.Flatten(name="flatten"),
    layers.Dropout(0.4, name="dropout"),
    layers.Dense(256, activation='relu', name="dense_feature"),
    layers.Dense(num_classes, activation='softmax', name="salida_probabilidades")
], name="CNN_Document_Classifier")

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

print("\n" + "=" * 70)
print(" 3. FASE DE ENTRENAMIENTO")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Guardar modelos en ambos formatos
model.save(PROJECT_DIR / "modelo_documentos.keras")
model.save(PROJECT_DIR / "modelo_documentos.h5")
print("\n[+] Modelo guardado en: modelo_documentos.keras y modelo_documentos.h5")

print("\n" + "=" * 70)
print(" 4. EVALUACIÓN EXHAUSTIVA SOBRE DATOS NUEVOS NO VISTOS (CONJUNTO TEST)")
print("=" * 70)

# Obtener etiquetas verdaderas y predicciones sobre el TEST SET
y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
predicciones_prob = model.predict(test_ds)
y_pred = np.argmax(predicciones_prob, axis=1)

# Cálculo de Métricas Clave
aciertos = np.sum(y_true == y_pred)
total_test = len(y_true)
tasa_acierto = (aciertos / total_test) * 100
tasa_error = (1.0 - (aciertos / total_test)) * 100

print(f"\n[*] Total de documentos de TEST (Nuevos / No Vistos): {total_test}")
print(f"[*] Documentos clasificados correctamente:            {aciertos}")
print(f"[*] TASA DE ACIERTO EN TEST:                           {tasa_acierto:.2f}%  (Meta requerida: > 98.5%)")
print(f"[*] TASA DE ERROR EN TEST:                             {tasa_error:.2f}%  (Meta requerida: < 3.0%)")

if tasa_acierto > 98.5 and tasa_error < 3.0:
    print("\n✅ ¡OBJETIVO CUMPLIDO EXITOSAMENTE! Métricas superan los requisitos de la rúbrica.")
else:
    print("\n⚠️ Métricas fuera de rango objetivo.")

print("\n--- REPORTE DE CLASIFICACIÓN EN DATOS NO VISTOS ---")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("--- MATRIZ DE CONFUSIÓN EN DATOS NO VISTOS ---")
print(cm)

print("\n" + "=" * 70)
print(" 4.5 EVALUACIÓN ADICIONAL SOBRE CONJUNTO TRAIN")
print("=" * 70)

# Predicción sobre train (datos que el modelo sí vio)
train_predicciones = model.predict(train_ds)
y_train_true = np.concatenate([y.numpy() for x, y in train_ds], axis=0)
y_train_pred = np.argmax(train_predicciones, axis=1)

train_aciertos = np.sum(y_train_true == y_train_pred)
train_total = len(y_train_true)
train_accuracy = (train_aciertos / train_total) * 100
train_error = 100.0 - train_accuracy

print(f"\n[*] Total documentos TRAIN: {train_total}")
print(f"[*] Aciertos en TRAIN:      {train_aciertos}")
print(f"[*] Accuracy en TRAIN:      {train_accuracy:.2f}%")
print(f"[*] Error en TRAIN:         {train_error:.2f}%")

# Estas métricas se incorporan al JSON al finalizar la generación de gráficas.
history_dict = {
    "train_accuracy": float(train_accuracy),
    "train_error": float(train_error)
}

print("\n" + "=" * 70)
print(" 5. GENERACIÓN DE GRÁFICAS DE TASAS DE ACIERTO Y ERROR")
print("=" * 70)

train_acc = np.array(history.history['accuracy']) * 100
val_acc = np.array(history.history['val_accuracy']) * 100
train_err = 100.0 - train_acc
val_err = 100.0 - val_acc
epochs_range = range(1, len(train_acc) + 1)

# Gráfica 1: Tasas de Acierto y Error
plt.figure(figsize=(14, 5))

# Subplot 1: Tasa de Acierto
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, 'b-o', label='Tasa Acierto Entrenamiento')
plt.plot(epochs_range, val_acc, 'g-s', label='Tasa Acierto Validación')
plt.axhline(y=98.5, color='r', linestyle='--', label='Meta Rúbrica (> 98.5%)')
plt.title('Evolución de la Tasa de Acierto (%)', fontsize=12, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Tasa de Acierto (%)')
plt.ylim(70, 102)
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)

# Subplot 2: Tasa de Error
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_err, 'r-o', label='Tasa Error Entrenamiento')
plt.plot(epochs_range, val_err, 'm-s', label='Tasa Error Validación')
plt.axhline(y=3.0, color='black', linestyle='--', label='Meta Rúbrica (< 3.0%)')
plt.title('Evolución de la Tasa de Error (%)', fontsize=12, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Tasa de Error (%)')
plt.ylim(-1, 30)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(PROJECT_DIR / "grafica_tasas_acierto_error.png", dpi=300)
plt.close()
print("[+] Gráfica guardada: grafica_tasas_acierto_error.png")

# Gráfica 2: Pérdida (Loss)
plt.figure(figsize=(7, 5))
plt.plot(epochs_range, history.history['loss'], 'b-o', label='Pérdida Entrenamiento')
plt.plot(epochs_range, history.history['val_loss'], 'r-s', label='Pérdida Validación')
plt.axhline(y=0.03, color='black', linestyle='--', label='Umbral Loss (0.03)')
plt.title('Evolución de la Pérdida (Loss)', fontsize=12, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Loss (Crossentropy)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(PROJECT_DIR / "grafica_perdida.png", dpi=300)
plt.close()
print("[+] Gráfica guardada: grafica_perdida.png")

# Guardar historial en JSON para el notebook
history_dict = {
    "accuracy": [float(x) for x in history.history['accuracy']],
    "val_accuracy": [float(x) for x in history.history['val_accuracy']],
    "loss": [float(x) for x in history.history['loss']],
    "val_loss": [float(x) for x in history.history['val_loss']],
    "train_accuracy": float(train_accuracy),
    "train_error": float(train_error),
    "test_accuracy": float(tasa_acierto),
    "test_error": float(tasa_error),
    "confusion_matrix": cm.tolist(),
    "class_names": class_names
}
with open(PROJECT_DIR / "training_metrics.json", "w", encoding="utf-8") as f:
    json.dump(history_dict, f, indent=2)

print("\n¡Entrenamiento y Evaluación Finalizados con Éxito!")
