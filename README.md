# PapaSalud v3 — interfaz renovada con el modelo original

Esta versión **no entrena otro modelo** y **no utiliza `modelo_v2.pkl`**. Mantiene el archivo original `modelo.pkl` (no incluido) y su clasificación de tres clases.

## Instalar en Windows / Visual Studio Code

1. Haz una copia de seguridad de tu carpeta `Aplicacion`.
2. Copia `app.py` y `requirements.txt` de este paquete a esa carpeta, reemplazando los anteriores. **No reemplaces ni borres `modelo.pkl`**.
3. Abre `Aplicacion` con VS Code, luego `Terminal > New Terminal`.
4. Instala bibliotecas (funciona aunque PowerShell bloquee la activación del entorno virtual):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si no existe `.venv`, créalo antes con `python -m venv .venv`.

5. Ejecuta:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

6. Abre `http://localhost:8501` y actualiza con Ctrl+F5 si veías la interfaz anterior. Detén el servidor con Ctrl+C.

## Historial y privacidad

- Los registros se guardan en `papasalud_historial.db` (se crea al guardar una evaluación).
- **No se guardan fotografías**; solo resultados, lote y observaciones.
- Para llevar registros a otro computador, descarga el CSV en la sección Historial.
- En **Streamlit Cloud**, SQLite es temporal y no sirve como historial permanente ni privado entre usuarios. Antes de usarla públicamente con datos reales, hay que integrar una base de datos persistente y autenticación.
- No subas `papasalud_historial.db` a GitHub. Se incluye un `.gitignore`.

## Publicar en Streamlit Cloud

Sube al repositorio `app.py`, `requirements.txt` y tu `modelo.pkl` original. Configura el archivo principal como `app.py`. No subas `.venv`, fotos privadas ni la base de datos local. Verifica que el modelo se pueda cargar en la versión de scikit-learn instalada en la nube.

## Interpretación

La puntuación por clase **no es el porcentaje de hoja enferma**. El modelo no valida si la imagen es de papa y las orientaciones no sustituyen una evaluación agronómica. Las estadísticas cuentan evaluaciones, no plantas enfermas en el lote.
