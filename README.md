# Simulacion-proyecto-11

## 📄 Sistema Inteligente de Clasificación y Extracción de Documentos (Document AI)

Sistema integral basado en **Visión por Computadora (Redes Neuronales Convolucionales - CNN)** y **Extracción Óptica de Información (OCR)** para la clasificación automática y digitalización estructurada de documentos empresariales en bases de datos relacionales **MySQL**.

---

### 🏛️ Arquitectura del Repositorio

El repositorio está organizado en dos módulos principales:

```text
Simulacion-proyecto-11/
├── Proyecto/                        # Módulo de Inteligencia Artificial & Entrenamiento
│   ├── RedesNeuronales_11.ipynb     # Notebook interactivo con todo el flujo de entrenamiento
│   ├── train_and_evaluate.py        # Script de entrenamiento CNN, validación y test
│   ├── prepare_dataset.py           # Pipeline de procesamiento y particionado del dataset
│   ├── generate_extra_plots.py      # Generador de gráficas de convergencia y métricas
│   ├── build_executed_notebook.py   # Constructor del notebook con celdas ejecutadas
│   ├── modelo_documentos.keras      # Modelo CNN entrenado (Keras 3)
│   ├── modelo_documentos.h5         # Modelo CNN en formato HDF5 legacy
│   ├── dataset.zip                  # Dataset estructurado en clases (9.3 MB)
│   ├── dataset_original.csv         # Metadatos del dataset original
│   ├── classes.json / classes.txt   # Mapeo de categorías
│   ├── training_metrics.json        # Métricas de entrenamiento y test (100% test accuracy)
│   └── *.png                        # Curvas de pérdida, acierto/error y matriz de confusión
│
├── Web/                             # Módulo del Sistema Web & Inferencia en Tiempo Real
│   ├── app.py                       # Servidor Flask, inferencia CNN, OCR y persistencia MySQL
│   ├── templates/
│   │   └── index.html               # Interfaz moderna (Tailwind CSS, Visor Split-Screen, Filtros)
│   ├── company-document-text.csv    # Índice de referencia documental para OCR
│   ├── requirements.txt             # Dependencias del servidor web
│   ├── iniciar_servidor_web.bat     # Script de inicio rápido con 1-clic
│   ├── modelo_documentos.keras      # Modelo de inferencia
│   └── documentos_de_prueba/        # Más de 400 documentos de muestra para pruebas
│
├── .gitignore                       # Filtro de exclusión para Git
└── README.md                        # Documentación técnica del proyecto
```

---

### 🎯 Tipos de Documentos Soportados

El sistema clasifica y extrae información especializada para 4 categorías empresariales:

1. **🧾 Facturas Comerciales (`invoices`)**:
   - Extracción de: N° Factura, Razón Social / Contacto, NIT / Customer ID, Fecha de Emisión, Dirección de Facturación, Total y Detalle de Líneas de Productos.
   - Tabla MySQL: `facturas`.

2. **📑 Órdenes de Compra (`PurchaseOrders`)**:
   - Extracción de: N° Orden, Cliente / Comprador, Fecha de Pedido, Cantidad de Items y Total Liquidado.
   - Tabla MySQL: `ordenes_compra`.

3. **🚚 Despachos de Envío (`Shipping orders`)**:
   - Extracción de: N° Guía de Despacho, Destinatario (*Ship To*), Transportadora (*Federal Shipping*, etc.), Fecha de Pedido, Fecha de Despacho, Dirección de Entrega y Total Flete.
   - Tabla MySQL: `despachos_envio`.

4. **📦 Reportes de Inventario (`Inventory Report`)**:
   - Extracción de: Identificador de Reporte, Periodo Mensual, Categoría del Almacén, Unidades Vendidas y Unidades en Stock.
   - Tabla MySQL: `reportes_inventario`.

---

### 🗄️ Esquema de Base de Datos MySQL (`docuai_sistema`)

El sistema cuenta con persistencia automática en **MySQL 8** mediante las siguientes tablas relacionales:

* `documentos_procesados`: Tabla maestra con el historial consolidado, predicción del modelo CNN, métricas de confianza, tiempo de procesamiento y enlace al archivo original.
* `facturas`: Tabla especializada para facturación.
* `ordenes_compra`: Tabla especializada para compras y adquisiciones.
* `despachos_envio`: Tabla especializada para logística y envíos.
* `reportes_inventario`: Tabla especializada para control de existencias en bodega.
* `items_detalle`: Tabla con el detalle desglosado de productos/artículos por cada documento mediante clave foránea (`documento_id`).

---

### 🚀 Instalación y Ejecución

#### 1. Requisitos Previos
* **Python 3.10 o 3.11**
* **MySQL Server** (Servicio local en puerto `3306`)
* Administrador de paquetes recomendado: **`uv`** o **`pip`**

#### 2. Configuración de Base de Datos MySQL
El sistema crea automáticamente la base de datos `docuai_sistema` y todas las tablas al iniciar la aplicación.
Configuración por defecto en `app.py`:
- **Host**: `127.0.0.1` (puerto `3306`)
- **Usuario**: definido mediante `MYSQL_USER` (por defecto: `udianasis`)
- **Contraseña**: definida mediante `MYSQL_PASSWORD`
- **Base de Datos**: `docuai_sistema`

En Windows PowerShell, configura las variables antes de iniciar la aplicación:

```powershell
$env:MYSQL_USER = "tu_usuario"
$env:MYSQL_PASSWORD = "tu_contraseña"
```

#### 3. Iniciar el Servidor Web

**Opción A: Inicio con script .bat (Windows)**
Haz doble clic en `Web/iniciar_servidor_web.bat`.

**Opción B: Inicio por consola**
```bash
cd Web
uv run --python 3.11 --with flask --with tensorflow --with pymupdf --with pillow --with winocr --with pymysql python app.py
```
O con `pip`:
```bash
cd Web
pip install -r requirements.txt
python app.py
```

Accede desde tu navegador a:
👉 **`http://localhost:5000`**

---

### 📊 Rendimiento del Modelo Convolucional (CNN)

* **Precisión en Test (Accuracy)**: **100.00%**
* **Tasa de Error**: **0.00%**
* **Tiempo Promedio de Inferencia**: **~0.02 - 0.25 segundos**
* **Matriz de Confusión**: Clasificación perfecta en el conjunto de prueba para todas las clases documentales.
