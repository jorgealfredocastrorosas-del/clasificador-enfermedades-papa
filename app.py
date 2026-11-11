from pathlib import Path
import joblib
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title='Salud de hojas de papa', page_icon='🥔', layout='centered')
CLASES = ['Sana', 'Tizón temprano', 'Tizón tardío']
MODELO = Path(__file__).resolve().parent / 'modelo.pkl'

@st.cache_resource(show_spinner=False)
def cargar_modelo():
    if not MODELO.exists():
        return None
    return joblib.load(MODELO)  # Cargar únicamente archivos de confianza.

def extraer_caracteristicas(imagen):
    img = imagen.convert('RGB').resize((32, 32))
    return np.asarray(img, dtype=np.float32).reshape(-1) / 255.0

st.markdown('''<style>
.block-container{max-width:850px;padding-top:2rem;padding-bottom:3rem}
.hero{background:linear-gradient(120deg,#164e38,#277456);padding:1.7rem;border-radius:18px;color:white;margin-bottom:1.5rem}
.hero h1{color:white;font-size:clamp(1.7rem,5vw,2.5rem);margin:0}
.hero p{color:#e2f4e9;margin:.6rem 0 0}
.result{padding:1.4rem;border-radius:15px;margin-top:1rem;border:1px solid}
.result h2{margin:0;font-size:1.65rem}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🥔 Salud de hojas de papa</h1><p>Clasificación orientativa con Random Forest · Sana · Tizón temprano · Tizón tardío</p></div>', unsafe_allow_html=True)
st.write('Sube una fotografía clara de una hoja de papa para obtener una clasificación orientativa.')

try:
    modelo = cargar_modelo()
except Exception:
    modelo = None
    st.error('No se pudo cargar modelo.pkl. Comprueba que se haya generado correctamente y que las versiones de las bibliotecas sean compatibles.')
if modelo is None:
    st.warning('Modelo no disponible. Primero ejecuta entrenar_modelo.py y coloca modelo.pkl junto a app.py. Después reinicia la aplicación.')

archivo = st.file_uploader('Selecciona una imagen de la hoja', type=['jpg','jpeg','png'], help='Archivos JPG, JPEG o PNG')
if archivo is not None:
    try:
        with Image.open(archivo) as original:
            imagen = original.convert('RGB')
        st.image(imagen, caption='Imagen cargada', use_container_width=True)
        if modelo is not None:
            vector = extraer_caracteristicas(imagen).reshape(1, -1)
            indice = int(modelo.predict(vector)[0])
            if indice not in (0, 1, 2):
                raise ValueError('El modelo devolvió una clase desconocida.')
            probabilidades = modelo.predict_proba(vector)[0]
            puntuacion = float(probabilidades[list(modelo.classes_).index(indice)])
            colores = {0:('#e8f7ec','#17613c'),1:('#fff2de','#9a560a'),2:('#ffe7e5','#a12b27')}
            fondo, texto = colores[indice]
            st.markdown(f'<div class="result" style="background:{fondo};color:{texto};border-color:{texto}"><small>RESULTADO DE LA CLASIFICACIÓN</small><h2>{CLASES[indice]}</h2><p>Puntuación del modelo: <b>{puntuacion:.1%}</b></p></div>', unsafe_allow_html=True)
            st.subheader('Puntuaciones por clase')
            for clase, prob in zip(modelo.classes_, probabilidades):
                st.write(f'{CLASES[int(clase)]}: {float(prob):.1%}')
                st.progress(float(prob))
            st.info('La puntuación del modelo no equivale a certeza diagnóstica. Este proyecto es académico y no sustituye una evaluación agronómica.')
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f'No se pudo procesar la imagen: {exc}')

st.caption('Modelo: RandomForestClassifier · Entrada: RGB 32 × 32 · PlantVillage')
