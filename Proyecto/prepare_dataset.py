import os
import shutil
import random
import pymupdf
import zipfile
from pathlib import Path

# Fijar semilla para reproducibilidad exacta
random.seed(42)

PROJECT_DIR = Path(__file__).resolve().parent
RAW_ROOT = PROJECT_DIR / "dataset_raw"
ZIP_SOURCE = PROJECT_DIR / "dataset.zip"
RAW_DIR = RAW_ROOT / "CompanyDocuments"
OUTPUT_DIR = PROJECT_DIR / "dataset"
CSV_SOURCE = RAW_ROOT / "company-document-text.csv"
CSV_DEST = PROJECT_DIR / "dataset_original.csv"

# El repositorio distribuye los PDF dentro de dataset.zip. Extraerlos solo
# cuando todavía no existe el directorio de origen esperado.
if not RAW_DIR.exists():
    if not ZIP_SOURCE.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {ZIP_SOURCE}. "
            "Coloca dataset.zip en la carpeta Proyecto."
        )
    print(f"[+] Extrayendo dataset desde {ZIP_SOURCE}...")
    with zipfile.ZipFile(ZIP_SOURCE) as archive:
        archive.extractall(RAW_ROOT)
    print(f"[+] Dataset extraído en {RAW_ROOT}")

# 1. Copiar CSV original requerido para la entrega
if CSV_SOURCE.exists():
    shutil.copy(CSV_SOURCE, CSV_DEST)
    print(f"[+] Dataset CSV copiado a {CSV_DEST}")

# Limpiar o crear carpetas de salida
for split in ["train", "val", "test"]:
    for cat in ["invoices", "PurchaseOrders", "Shipping orders", "Inventory Report"]:
        (OUTPUT_DIR / split / cat).mkdir(parents=True, exist_ok=True)

# 2. Mapear todos los PDFs recursivamente por categoría
categories = ["invoices", "PurchaseOrders", "Shipping orders", "Inventory Report"]
all_pdfs = {cat: [] for cat in categories}

for cat in categories:
    cat_path = RAW_DIR / cat
    for root, _, files in os.walk(cat_path):
        for f in files:
            if f.lower().endswith(".pdf"):
                all_pdfs[cat].append(Path(root) / f)

print("\n--- Recuento de PDFs encontrados ---")
for cat, files in all_pdfs.items():
    print(f"  {cat}: {len(files)} PDFs")

# 3. Dividir y convertir a imagen
# 70% Train, 15% Val, 15% Test (Datos No Vistos)
print("\n--- Convirtiendo PDFs a Imágenes y Dividiendo Dataset ---")
split_counts = {"train": 0, "val": 0, "test": 0}

for cat, files in all_pdfs.items():
    random.shuffle(files)
    n = len(files)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    # El resto va a test (aprox 15%)
    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:]
    }

    for split_name, split_files in splits.items():
        dest_dir = OUTPUT_DIR / split_name / cat
        for pdf_path in split_files:
            img_name = pdf_path.stem + ".jpg"
            dest_img = dest_dir / img_name
            if not dest_img.exists():
                try:
                    doc = pymupdf.open(pdf_path)
                    page = doc[0]
                    # Renderizar a 120 DPI para excelente calidad de texto y velocidad
                    pix = page.get_pixmap(dpi=120)
                    pix.save(str(dest_img))
                    doc.close()
                except Exception as e:
                    print(f"Error convirtiendo {pdf_path}: {e}")
            split_counts[split_name] += 1
        print(f"  [{cat}] -> {split_name}: {len(split_files)} imágenes")

print("\n--- Resumen Final de Partición ---")
print(f"  Train (Entrenamiento):        {split_counts['train']} imágenes (70%)")
print(f"  Val (Validación):             {split_counts['val']} imágenes (15%)")
print(f"  Test (Datos Nuevos No Vistos): {split_counts['test']} imágenes (15%)")
print(f"  Total:                        {sum(split_counts.values())} imágenes")
print("\n¡Dataset estructurado con éxito!")
