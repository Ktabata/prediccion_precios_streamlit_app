import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import datetime # Para obtener el año actual si se usa como referencia

# --- Configuración de la Página de Streamlit (DEBE SER EL PRIMER COMANDO DE STREAMLIT) ---
st.set_page_config(page_title="Predicción de Precio de Viviendas", 
                   page_icon="🏡", 
                   layout="wide")

# --- Constantes y Configuración ---
#MODEL_PACKAGE_FILE = 'price_prediction_package_v1.pkl' # Nombre de tu archivo .pkl guardado

import os
# ... otras importaciones ...

# --- Construir la ruta al archivo del modelo de forma robusta ---
# Obtiene la ruta del directorio donde se encuentra este script (app_streamlit.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Nombre del archivo del modelo
MODEL_FILE_NAME = 'price_prediction_package_v1.pkl'

# Une las dos partes para crear la ruta completa y correcta
MODEL_PACKAGE_FILE = os.path.join(BASE_DIR, MODEL_FILE_NAME)

# --- Funciones de Carga de Artefactos (con caché de Streamlit) ---
@st.cache_resource # Cache para objetos que no deben ser serializados por st.cache_data (modelos, encoders, scalers)
def load_model_package(path):
    """Carga el paquete completo del modelo desde un archivo .pkl."""
    if os.path.exists(path):
        with open(path, 'rb') as file:
            package = pickle.load(file)
        return package
    else:
        # Esta llamada a st.error() se ejecutará DESPUÉS de set_page_config si load_model_package se llama más tarde.
        # Si se llama antes, moverla o manejar el error de otra forma.
        # Por ahora, la carga se hace después de set_page_config, así que está bien.
        st.error(f"Error Crítico: El archivo del paquete del modelo '{path}' no fue encontrado.")
        return None

# --- Cargar el Paquete del Modelo ---
loaded_package = load_model_package(MODEL_PACKAGE_FILE)

# Inicializar variables desde el paquete (con valores por defecto si el paquete no se carga)
model = None
scaler = None
city_encoder = None
model_columns = []
city_options = []
PREDICTION_YEAR = 2014 # Default si no se carga del paquete
TARGET_COLUMN_NAME = 'price_log' # Default

if loaded_package:
    model = loaded_package.get('model')
    scaler = loaded_package.get('scaler')
    city_encoder = loaded_package.get('city_encoder')
    model_columns = loaded_package.get('model_columns_ordered', [])
    city_options = loaded_package.get('city_options_for_dropdown', [])
    
    if 'feature_engineering_details' in loaded_package and \
       'prediction_year_reference' in loaded_package['feature_engineering_details']:
        PREDICTION_YEAR = loaded_package['feature_engineering_details']['prediction_year_reference']
    
    if 'target_variable' in loaded_package:
        TARGET_COLUMN_NAME = loaded_package['target_variable']

    if not all([model, scaler, city_encoder, model_columns]):
        st.error("Error Crítico: Uno o más artefactos esenciales no se cargaron correctamente del paquete. La aplicación podría no funcionar.")
    else:
        st.success(f"Paquete del modelo '{MODEL_PACKAGE_FILE}' cargado exitosamente.")
        print(f"Debug: PREDICTION_YEAR={PREDICTION_YEAR}, TARGET_COLUMN_NAME={TARGET_COLUMN_NAME}") # Para depuración en consola
else:
    # El error ya se muestra en load_model_package si el archivo no existe.
    # Si loaded_package es None por otra razón, podríamos necesitar un st.error() aquí.
    if not os.path.exists(MODEL_PACKAGE_FILE): # Para evitar doble error si ya se mostró
        pass
    elif not loaded_package: # Si es None pero el archivo existía, indica otro problema de carga.
        st.error("Error: No se pudo cargar el paquete del modelo por una razón desconocida.")


# --- Título de la Aplicación ---
st.title("🏡 Estimador de Precio de Viviendas")
st.markdown("Ingrese las características de la vivienda para obtener una estimación de su precio de mercado.")
st.markdown(f"*(Las predicciones utilizan el año de referencia: **{PREDICTION_YEAR}**)*")

# --- Entradas del Usuario en la Barra Lateral ---
st.sidebar.header("Características de la Vivienda:")

# Usar un formulario para agrupar entradas y tener un botón de envío
with st.sidebar.form(key='house_features_form'):
    # Valores por defecto para una casa típica
    default_bedrooms = 3
    default_bathrooms = 2.0 # Permitir float para baños
    default_sqft_living = 1800
    default_sqft_lot = 5500
    default_floors = 1.0 # Permitir float para pisos
    default_yr_built = 1985
    default_city = "Seattle" if city_options and "Seattle" in city_options else (city_options[0] if city_options else "Desconocida")
    default_sale_month = 6 # Junio

    # Columnas del formulario
    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input("Habitaciones", min_value=0, max_value=15, value=default_bedrooms, step=1)
        sqft_living = st.number_input("Pies Cuadrados Habitables (sqft)", min_value=200, max_value=15000, value=default_sqft_living, step=10)
        floors = st.number_input("Pisos", min_value=1.0, max_value=4.0, value=default_floors, step=0.5)
        view = st.slider("Calidad de la Vista (0-4)", min_value=0, max_value=4, value=0)
        yr_built = st.number_input("Año de Construcción", min_value=1800, max_value=PREDICTION_YEAR, value=default_yr_built, step=1)
        city_idx = 0
        if city_options: # Solo intentar encontrar el índice si city_options no está vacío
            try:
                city_idx = city_options.index(default_city)
            except ValueError: # Si default_city (ej. "Desconocida") no está en city_options
                city_idx = 0 # Usar el primer elemento
        
        city = st.selectbox("Ciudad", options=sorted(city_options), 
                            index=city_idx)
        
    with col2:
        bathrooms = st.number_input("Baños", min_value=0.0, max_value=10.0, value=default_bathrooms, step=0.25, format="%.2f")
        sqft_lot = st.number_input("Pies Cuadrados del Lote (sqft)", min_value=300, max_value=1200000, value=default_sqft_lot, step=100)
        
        waterfront_options_map = {"No": 0, "Sí": 1}
        waterfront_selected = st.selectbox("Frente al Mar", options=list(waterfront_options_map.keys()))
        waterfront = waterfront_options_map[waterfront_selected]
        
        condition = st.slider("Condición de la Propiedad (1-5)", min_value=1, max_value=5, value=3)
        yr_renovated = st.number_input("Año de Renovación (0 si no aplica)", min_value=0, max_value=PREDICTION_YEAR, value=0, step=1)
        sale_month = st.selectbox("Mes de Venta (Estimado)", options=list(range(1, 13)), index=default_sale_month - 1)

    # Inputs que podrían depender de otros, o son menos comunes (opcionalmente en otra sección)
    st.markdown("---") # Separador
    sqft_above = st.number_input("Pies Cuadrados Sobre Nivel (sqft_above)", min_value=200, max_value=10000, value=int(default_sqft_living * 0.8), step=10)
    sqft_basement = st.number_input("Pies Cuadrados Sótano (sqft_basement, 0 si no tiene)", min_value=0, max_value=5000, value=int(default_sqft_living * 0.2), step=10)

    submit_button = st.form_submit_button(label="📊 Estimar Precio")


# --- Lógica de Predicción y Visualización de Resultados ---
if submit_button:
    if not all([model, scaler, city_encoder, model_columns]):
        st.error("La aplicación no puede realizar predicciones porque los artefactos del modelo no se cargaron correctamente.")
    else:
        try:
            # 1. Recopilar datos del formulario en un diccionario
            features_input = {
                'bedrooms': float(bedrooms),
                'bathrooms': float(bathrooms),
                'sqft_living': int(sqft_living),
                'sqft_lot': int(sqft_lot),
                'floors': float(floors),
                'waterfront': int(waterfront),
                'view': int(view),
                'condition': int(condition),
                'sqft_above': int(sqft_above),
                'sqft_basement': int(sqft_basement),
                'yr_built': int(yr_built),
                # 'yr_renovated' se usa para derivar, no directamente si no está en model_columns
                'city': city, # Se codificará
                'sale_month': int(sale_month),
            }

            # 2. Ingeniería de características (consistente con el entrenamiento)
            features_input['age_at_sale'] = PREDICTION_YEAR - features_input['yr_built']
            # Lógica para yr_renovated y características derivadas
            current_yr_renovated = int(yr_renovated) # Asegurar que yr_renovated es int
            if current_yr_renovated > 0 and current_yr_renovated <= PREDICTION_YEAR:
                features_input['yrs_since_renovation'] = PREDICTION_YEAR - current_yr_renovated
                features_input['was_renovated'] = 1
            else:
                features_input['yrs_since_renovation'] = features_input['age_at_sale'] # O 0, según tu lógica de entrenamiento
                features_input['was_renovated'] = 0
            
            # Crear DataFrame para preprocesamiento
            input_df = pd.DataFrame([features_input])

            # 3. Preprocesamiento: One-Hot Encode 'city'
            city_to_encode_df = input_df[['city']]
            
            # Columnas numéricas y derivadas (excluyendo 'city' y 'yr_renovated' si no es feature directa)
            numeric_and_derived_cols = [
                'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 
                'yr_built', 'sale_month', 'age_at_sale', 'yrs_since_renovation', 
                'was_renovated'
            ]
            # Tomar solo las columnas que existen en input_df (para evitar errores si alguna no se incluyó)
            other_features_df = input_df[[col for col in numeric_and_derived_cols if col in input_df.columns]]

            city_encoded_array = city_encoder.transform(city_to_encode_df)
            
            # Obtener nombres de características de OHE
            # Usar city_encoder.categories_[0] es más robusto si get_feature_names_out no está en la versión de pickle
            try:
                ohe_feature_names = city_encoder.get_feature_names_out(['city'])
            except AttributeError: # Fallback para versiones antiguas de scikit-learn
                ohe_feature_names = [f"city_{cat}" for cat in city_encoder.categories_[0]]
            
            city_encoded_df = pd.DataFrame(city_encoded_array, columns=ohe_feature_names, index=other_features_df.index)

            # Combinar
            processed_df_unordered = pd.concat([other_features_df, city_encoded_df], axis=1)
            
            # 4. Asegurar orden y presencia de todas las columnas del modelo
            # Crear un DataFrame base con todas las columnas esperadas por el modelo, inicializadas con 0.0
            # y con el tipo de dato correcto para evitar problemas de inferencia con fillna o concat.
            final_input_features_dict = {}
            for col in model_columns:
                if col in processed_df_unordered.columns:
                    # Tomar el tipo de la columna en processed_df_unordered si existe
                    # y si es numérico, si no, float. OHE son float.
                    dtype = processed_df_unordered[col].dtype
                    if pd.api.types.is_numeric_dtype(dtype):
                         final_input_features_dict[col] = pd.Series(dtype=dtype)
                    else: # Debería ser OHE, que es float
                         final_input_features_dict[col] = pd.Series(dtype=float)
                else: # Columnas OHE que no están presentes para la ciudad seleccionada
                    final_input_features_dict[col] = pd.Series(dtype=float)

            final_input_features = pd.DataFrame(final_input_features_dict)
            # Llenar con los valores de processed_df_unordered. Las columnas que no estén se quedarán como NaN.
            final_input_features = pd.concat([final_input_features, processed_df_unordered], ignore_index=True)
            final_input_features = final_input_features.fillna(0.0) # Rellenar NaNs con 0.0 (para OHE)
            final_input_features = final_input_features[model_columns].astype(float) # Asegurar orden y tipo float

            # 5. Escalar las características
            scaled_features_array = scaler.transform(final_input_features)

            # 6. Realizar la predicción
            prediction_log = model.predict(scaled_features_array)
            
            # 7. Revertir la transformación logarítmica (si aplica)
            if TARGET_COLUMN_NAME == 'price_log':
                prediction_original_scale = np.expm1(prediction_log[0])
            else:
                prediction_original_scale = prediction_log[0] # Ya está en escala original

            # Mostrar el resultado
            st.subheader("Resultado de la Predicción:")
            st.metric(label="Precio Estimado de la Vivienda", value=f"${prediction_original_scale:,.2f}")
            
            # Mostrar datos de entrada y procesados (opcional)
            with st.expander("Detalles de la Entrada y Procesamiento"):
                st.write("**Características Ingresadas:**", features_input)
                st.write("**DataFrame Final Enviado al Modelo (antes de escalar):**")
                st.dataframe(final_input_features.style.format("{:.2f}"))

        except Exception as e:
            st.error(f"Ocurrió un error durante el proceso de predicción:")
            st.exception(e) # Muestra el traceback completo en la app Streamlit para depuración

# --- Información Adicional o Pie de Página (Opcional) ---
st.sidebar.markdown("---")
st.sidebar.info(
    "Esta es una aplicación de demostración para predecir precios de viviendas "
    "basada en un modelo XGBoost. Los resultados son estimaciones."
)
