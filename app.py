"""PapaSalud v3: usa exclusivamente el RandomForest original modelo.pkl."""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import io
import sqlite3
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / 'modelo.pkl'
DB_PATH = BASE / 'papasalud_historial.db'
CLASES = {0: 'Sana', 1: 'Tizón temprano', 2: 'Tizón tardío'}
TONOS = {0: ('#e6f5ed','#146c43'), 1: ('#fff1dc','#9a570c'), 2: ('#ffebe8','#ad302a')}
st.set_page_config(page_title='PapaSalud | Cultivo de papa', page_icon='🌿', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp{background:#f3f7f4;color:#18352a}.block-container{max-width:1180px;padding-top:1.5rem;padding-bottom:3rem}
.hero{background:linear-gradient(115deg,#123e30,#267f59);color:white;border-radius:22px;padding:2.1rem 2.3rem;margin-bottom:1.5rem;box-shadow:0 12px 30px #0a3d251a}
.hero h1{color:white;font-size:clamp(1.9rem,4vw,3.1rem);margin:0 0 .6rem}.hero p{color:#e2f5e9;margin:0;font-size:1.05rem}
.panel{background:white;border:1px solid #dce9df;border-radius:18px;padding:1.3rem;margin:.8rem 0;box-shadow:0 6px 18px #0a3d250d}
.result{border-radius:17px;padding:1.4rem;margin:1rem 0;border:1px solid}.result h2{margin:.3rem 0;font-size:2rem;color:inherit}
[data-testid="stMetric"]{background:white;border:1px solid #dce9df;padding:1rem;border-radius:14px}
@media(max-width:640px){.hero{padding:1.3rem}.block-container{padding-top:.8rem}.result h2{font-size:1.5rem}}
</style>''', unsafe_allow_html=True)

@st.cache_resource(show_spinner='Cargando modelo original...')
def cargar_modelo():
    return joblib.load(MODEL_PATH)  # Cargar únicamente archivos propios y confiables.

def vectorizar(imagen):
    # Mismo preprocesamiento que el modelo original: RGB, 32x32, aplanado, /255.
    return np.asarray(imagen.convert('RGB').resize((32, 32)), dtype=np.float32).reshape(1, -1) / 255.0

def abrir_imagen(archivo):
    datos = archivo.getvalue()
    if len(datos) > 12 * 1024 * 1024:
        raise ValueError('La fotografía supera 12 MB.')
    with Image.open(io.BytesIO(datos)) as im:
        im.verify()
    with Image.open(io.BytesIO(datos)) as im:
        im = ImageOps.exif_transpose(im)
        if im.width < 64 or im.height < 64:
            raise ValueError('La fotografía es demasiado pequeña.')
        return im.convert('RGB').copy()

def conectar():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute('''CREATE TABLE IF NOT EXISTS evaluaciones (
        id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT NOT NULL, lote TEXT NOT NULL,
        resultado TEXT NOT NULL, puntuacion REAL NOT NULL, sana REAL NOT NULL,
        temprano REAL NOT NULL, tardio REAL NOT NULL, otras_plantas TEXT,
        humedad TEXT, evolucion TEXT, observaciones TEXT, huella TEXT NOT NULL)''')
    con.commit()
    return con

def guardar(registro):
    columnas = ['fecha','lote','resultado','puntuacion','sana','temprano','tardio','otras_plantas','humedad','evolucion','observaciones','huella']
    with conectar() as con:
        con.execute(f"INSERT INTO evaluaciones ({','.join(columnas)}) VALUES ({','.join('?' for _ in columnas)})", [registro[c] for c in columnas])

def historial():
    with conectar() as con:
        return pd.read_sql_query('SELECT id,fecha,lote,resultado,puntuacion,sana,temprano,tardio,otras_plantas,humedad,evolucion,observaciones FROM evaluaciones ORDER BY id DESC', con)

def hero(titulo, descripcion):
    st.markdown(f'<div class="hero"><h1>{html.escape(titulo)}</h1><p>{html.escape(descripcion)}</p></div>', unsafe_allow_html=True)

def orientaciones(clase, otras, humedad, evolucion):
    base = {
        0: ['Continuar el monitoreo periódico del cultivo.', 'Observar hojas de distintas zonas del lote; una sola fotografía no representa todo el cultivo.'],
        1: ['Inspeccionar otras hojas y plantas cercanas para comprobar si hay síntomas similares.', 'Registrar la ubicación de los síntomas y solicitar confirmación agronómica antes de decidir un tratamiento.'],
        2: ['Priorizar la inspección del sector donde se tomó la fotografía.', 'Solicitar una valoración agronómica oportuna: el tizón tardío puede avanzar rápidamente bajo condiciones favorables.'],
    }[clase].copy()
    if otras == 'Sí':
        base.append('Como observas síntomas en otras plantas, registra cuáles sectores están afectados y solicita una revisión del lote.')
    if humedad == 'Alta o lluvias recientes':
        base.append('La humedad o las lluvias recientes justifican reforzar la observación y consultar condiciones de riesgo con asistencia técnica.')
    if evolucion == 'Se ha extendido rápidamente':
        base.append('Por la evolución que reportas, busca orientación agronómica con prontitud.')
    return base

try:
    modelo = cargar_modelo() if MODEL_PATH.exists() else None
    if modelo is not None and (not hasattr(modelo, 'predict_proba') or set(map(int, modelo.classes_)) != {0,1,2}):
        raise ValueError('El archivo no contiene exactamente las clases 0, 1 y 2 del modelo original.')
    error_modelo = None
except Exception as exc:
    modelo, error_modelo = None, str(exc)

with st.sidebar:
    st.markdown('## 🌿 PapaSalud')
    st.caption('Asistente de observación del cultivo de papa')
    pagina = st.radio('Navegación', ['🏠 Inicio','📷 Analizar hoja','📚 Historial y estadísticas','ℹ️ Acerca del proyecto'], label_visibility='collapsed')
    st.divider()
    if modelo is None:
        st.error('No se pudo cargar modelo.pkl')
    else:
        st.success('Modelo original disponible · 3 clases')
    st.caption('Prototipo académico. No sustituye una evaluación agronómica.')

if pagina == '🏠 Inicio':
    hero('🌱 PapaSalud', 'Tecnología de apoyo para observar hojas de papa y organizar evaluaciones por lote.')
    a,b,c = st.columns(3)
    a.metric('Clases del modelo','3')
    b.metric('Preprocesamiento','RGB 32 × 32')
    try:
        c.metric('Evaluaciones registradas',len(historial()))
    except sqlite3.Error:
        c.metric('Evaluaciones registradas','—')
    st.markdown('### ¿Qué puedes hacer?')
    st.markdown('''<div class="panel"><b>📷 Analizar una hoja:</b> sube una fotografía o utiliza la cámara del celular.<br><br>
<b>🌾 Contextualizar:</b> responde preguntas sencillas sobre el lote y recibe orientaciones de inspección.<br><br>
<b>📚 Registrar:</b> guarda voluntariamente los resultados, consulta gráficos y exporta datos en CSV.</div>''',unsafe_allow_html=True)
    st.warning('La aplicación presupone que la imagen es de una hoja de papa. No comprueba la especie ni mide qué porcentaje de la hoja está enfermo.')

elif pagina == '📷 Analizar hoja':
    hero('📷 Análisis de hojas', 'Carga una hoja de papa y consulta la clasificación orientativa del modelo original.')
    if modelo is None:
        st.error(f'Coloca el archivo original modelo.pkl junto a app.py. {error_modelo or ""}')
        st.stop()
    modo = st.radio('Origen de la fotografía',['Subir imagen','Usar cámara'],horizontal=True)
    archivo = st.file_uploader('Imagen JPG, JPEG o PNG',type=['jpg','jpeg','png']) if modo == 'Subir imagen' else st.camera_input('Fotografiar hoja')
    if archivo is not None:
        try:
            imagen = abrir_imagen(archivo)
            probabilidades = modelo.predict_proba(vectorizar(imagen))[0]
            puntos = {int(c):float(p) for c,p in zip(modelo.classes_,probabilidades)}
            clase = max(puntos,key=puntos.get)
            fondo,color = TONOS[clase]
            izq,der = st.columns([1,1.15],gap='large')
            with izq:
                st.image(imagen,caption='Fotografía seleccionada',use_container_width=True)
            with der:
                st.markdown(f'<div class="result" style="background:{fondo};color:{color};border-color:{color}"><small>CLASIFICACIÓN ORIENTATIVA</small><h2>{CLASES[clase]}</h2><p>Puntuación de esta clase: <b>{puntos[clase]:.1%}</b></p></div>',unsafe_allow_html=True)
                st.markdown('**Puntuaciones por clase**')
                for i in range(3):
                    st.write(f'{CLASES[i]}: {puntos[i]:.1%}')
                    st.progress(puntos[i])
                st.caption('Estas puntuaciones NO representan el porcentaje de tejido afectado ni certeza diagnóstica.')
            st.divider()
            st.subheader('🌾 Contexto del cultivo')
            lote = st.text_input('Nombre o código del lote',placeholder='Ejemplo: Lote 1 - Vereda El Encano',max_chars=100)
            q1,q2,q3 = st.columns(3)
            with q1: otras = st.selectbox('¿Hay síntomas en otras plantas?',['No sé','No','Sí'])
            with q2: humedad = st.selectbox('¿Cómo ha estado el clima?',['No sé','Seco','Alta o lluvias recientes'])
            with q3: evolucion = st.selectbox('¿Cómo han evolucionado los síntomas?',['No sé','Sin cambios','Se ha extendido rápidamente'])
            observaciones = st.text_area('Observaciones adicionales (opcional)',max_chars=500,placeholder='Describe lo observado, sin incluir datos personales.')
            st.subheader('💡 Orientaciones para la inspección')
            for texto in orientaciones(clase,otras,humedad,evolucion):
                st.markdown(f'- {texto}')
            st.info('Las orientaciones se basan en reglas generales y tus respuestas; no son un diagnóstico ni una recomendación de dosis o aplicación de fungicidas.')
            st.divider()
            st.subheader('💾 Guardar evaluación')
            st.caption('El registro guarda fecha, lote, resultados y respuestas. No almacena la fotografía. Evita introducir datos personales.')
            if st.button('Guardar esta evaluación',type='primary'):
                if not lote.strip():
                    st.warning('Escribe primero un nombre o código de lote.')
                else:
                    try:
                        guardar({'fecha':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'lote':lote.strip(),'resultado':CLASES[clase],
                                'puntuacion':round(puntos[clase]*100,2),'sana':round(puntos[0]*100,2),
                                'temprano':round(puntos[1]*100,2),'tardio':round(puntos[2]*100,2),
                                'otras_plantas':otras,'humedad':humedad,'evolucion':evolucion,
                                'observaciones':observaciones.strip(), 'huella':hashlib.sha256(archivo.getvalue()).hexdigest()[:16]})
                        st.success('Evaluación guardada. Puedes consultarla en Historial y estadísticas.')
                    except sqlite3.Error as exc:
                        st.error(f'No se pudo guardar el registro: {exc}')
        except (OSError,ValueError,UnidentifiedImageError) as exc:
            st.error(f'No fue posible procesar la imagen: {exc}')

elif pagina == '📚 Historial y estadísticas':
    hero('📚 Historial y estadísticas','Consulta registros de evaluaciones y exporta la información para tu proyecto.')
    try:
        df = historial()
    except sqlite3.Error as exc:
        st.error(f'No se pudo abrir el historial: {exc}')
        st.stop()
    if df.empty:
        st.info('Todavía no has guardado evaluaciones. Ve a Analizar hoja y registra la primera.')
    else:
        lotes = sorted(df['lote'].unique().tolist())
        seleccion = st.selectbox('Filtrar por lote',['Todos']+lotes)
        vista = df if seleccion=='Todos' else df[df['lote']==seleccion]
        c1,c2,c3 = st.columns(3)
        c1.metric('Evaluaciones',len(vista))
        c2.metric('Clasificadas como sanas',int((vista['resultado']=='Sana').sum()))
        c3.metric('Clasificadas con tizón',int(vista['resultado'].isin(['Tizón temprano','Tizón tardío']).sum()))
        st.markdown('### Distribución de clasificaciones')
        conteos = vista['resultado'].value_counts().reindex(list(CLASES.values()),fill_value=0)
        st.bar_chart(conteos)
        st.caption('Los gráficos muestran conteos de fotografías clasificadas, no prevalencia ni gravedad de enfermedad en el lote.')
        st.dataframe(vista.drop(columns=['id']),use_container_width=True,hide_index=True)
        st.download_button('⬇️ Descargar registros CSV',data=vista.drop(columns=['id']).to_csv(index=False).encode('utf-8-sig'),file_name='papasalud_registros.csv',mime='text/csv')
    st.warning('Los registros se guardan en una base SQLite local. En Streamlit Cloud el almacenamiento puede perderse al reiniciar o redesplegar la aplicación y puede compartirse entre visitantes. Para uso real con agricultores se requiere base de datos persistente y controles de acceso.')

else:
    hero('ℹ️ Acerca de PapaSalud','Alcance y limitaciones del prototipo académico.')
    st.markdown('''### Modelo utilizado
Se utiliza exclusivamente el **RandomForestClassifier original** guardado en `modelo.pkl` con las clases **Sana, Tizón temprano y Tizón tardío**.

### Procesamiento
Cada imagen se convierte a RGB, se redimensiona a **32 × 32**, se aplana y se normaliza dividiendo sus valores entre **255.0**.

### Qué hace
Clasifica fotografías que el usuario presenta como hojas de papa; muestra puntuaciones, orientaciones generales y permite guardar resultados por lote.

### Qué no hace
No valida si la fotografía es de papa, no determina la severidad ni el área afectada, no diagnostica definitivamente, no indica dosis de fungicidas y no calcula ahorros económicos reales. Las orientaciones no sustituyen una visita agronómica.

### Uso responsable
Evita registrar datos personales. Antes de tomar decisiones de manejo en campo, confirma las observaciones con personal técnico y valida el modelo con fotografías reales del entorno agrícola.''')

st.caption('PapaSalud v3 · Modelo original · Random Forest · RGB 32 × 32 · Prototipo académico')
