# Data Science Project Boilerplate

This boilerplate is designed to kickstart data science projects by providing a basic setup for database connections, data processing, and machine learning model development. It includes a structured folder organization for your datasets and a set of pre-defined Python packages necessary for most data science tasks.

## Structure

The project is organized as follows:

- **`src/app.py`** → Main Python script where your project will run.
- **`src/explore.ipynb`** → Notebook for exploration and testing. Once exploration is complete, migrate the clean code to `app.py`.
- **`src/utils.py`** → Auxiliary functions, such as database connection.
- **`requirements.txt`** → List of required Python packages.
- **`models/`** → Will contain your SQLAlchemy model classes.
- **`data/`** → Stores datasets at different stages:
  - **`data/raw/`** → Raw data.
  - **`data/interim/`** → Temporarily transformed data.
  - **`data/processed/`** → Data ready for analysis.


## ⚡ Initial Setup in Codespaces (Recommended)

No manual setup is required, as **Codespaces is automatically configured** with the predefined files created by the academy for you. Just follow these steps:

1. **Wait for the environment to configure automatically**.
   - All necessary packages and the database will install themselves.
   - The automatically created `username` and `db_name` are in the **`.env`** file at the root of the project.
2. **Once Codespaces is ready, you can start working immediately**.


## 💻 Local Setup (Only if you can't use Codespaces)

**Prerequisites**

Make sure you have Python 3.11+ installed on your machine. You will also need pip to install the Python packages.

**Installation**

Clone the project repository to your local machine.

Navigate to the project directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

**Create a database (if necessary)**

Create a new database within the Postgres engine by customizing and executing the following command:

```bash
$ psql -U postgres -c "DO \$\$ BEGIN 
    CREATE USER my_user WITH PASSWORD 'my_password'; 
    CREATE DATABASE my_database OWNER my_user; 
END \$\$;"
```
Connect to the Postgres engine to use your database, manipulate tables, and data:

```bash
$ psql -U my_user -d my_database
```

Once inside PSQL, you can create tables, run queries, insert, update, or delete data, and much more!

**Environment Variables**

Create a .env file in the root directory of the project to store your environment variables, such as your database connection string:

```makefile
DATABASE_URL="postgresql://<USER>:<PASSWORD>@<HOST>:<PORT>/<DB_NAME>"

#example
DATABASE_URL="postgresql://my_user:my_password@localhost:5432/my_database"
```

## Running the Application

To run the application, execute the app.py script from the root directory of the project:

```bash
python src/app.py
```

## Adding Models

To add SQLAlchemy model classes, create new Python script files within the models/ directory. These classes should be defined according to your database schema.

Example model definition (`models/example_model.py`):

```py
from sqlalchemy.orm import declarative_base
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

Base = declarative_base()

class ExampleModel(Base):
    __tablename__ = 'example_table'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
```

## Working with Data

You can place your raw datasets in the data/raw directory, intermediate datasets in data/interim, and processed datasets ready for analysis in data/processed.

To process data, you can modify the app.py script to include your data processing steps, using pandas for data manipulation and analysis.

## Contributors

This template was built as part of the [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) by 4Geeks Academy by [Alejandro Sanchez](https://twitter.com/alesanchezr) and many other contributors. Learn more about [4Geeks Academy BootCamp programs](https://4geeksacademy.com/us/programs) here.

Other templates and resources like this can be found on the school's GitHub page.

Proyecto de Predicción de Precios de Viviendas
Este proyecto implementa un modelo de machine learning para predecir precios de viviendas y lo despliega como una aplicación web interactiva usando Streamlit. El ciclo de vida completo del proyecto, desde el tratamiento de datos brutos hasta el despliegue final, está documentado en este repositorio.

🚀 Demo en Vivo
La aplicación está desplegada en Streamlit Community Cloud y puedes probarla aquí:

➡️ Probar la Aplicación de Predicción de Precios ⬅️

(https://prediccionpreciosappapp-ktabata.streamlit.app/)

📋 Descripción del Proyecto
El objetivo de este proyecto fue construir un modelo de regresión robusto para estimar el valor de mercado de propiedades inmobiliarias basado en sus características. El proceso abarcó los siguientes pasos clave:

Tratamiento de Datos: El proyecto comenzó con un desafío considerable: un conjunto de datos en formato PDF con inconsistencias, errores de formato y datos mal alineados. Se realizó un proceso exhaustivo de extracción, limpieza y preprocesamiento para transformar estos datos en un formato estructurado y utilizable.

Análisis Exploratorio de Datos (EDA): Se analizaron las distribuciones de las variables, las correlaciones entre ellas y su relación con el precio de la vivienda para obtener insights iniciales.

Ingeniería de Características: Se crearon nuevas características para mejorar el poder predictivo del modelo, tales como:

age_at_sale: Antigüedad de la casa al momento de la venta.

yrs_since_renovation: Años desde la última renovación.

was_renovated: Indicador binario de si la propiedad fue renovada.

La variable objetivo price fue transformada a price_log para manejar su distribución sesgada.

Preprocesamiento y Modelado:

Las variables categóricas como city fueron codificadas usando One-Hot Encoding.

Las características numéricas fueron estandarizadas con StandardScaler.

Se entrenaron y evaluaron múltiples modelos, incluyendo Regresión Lineal, Random Forest (con optimización de hiperparámetros) y XGBoost.

Selección y Evaluación del Modelo: El modelo XGBoost demostró ser el más performante, obteniendo un R² de 0.68 y un Error Absoluto Medio (MAE) de ~$125,839 en el conjunto de prueba. El análisis de residuos confirmó que el modelo es robusto y no presenta sesgos obvios.

Despliegue: El modelo final, junto con sus preprocesadores (escalador y codificador), fue empaquetado en un único archivo .pkl y desplegado como una aplicación web interactiva utilizando Streamlit y Streamlit Community Cloud.

✨ Características Principales
Interfaz Interactiva: Permite a los usuarios ingresar las características de una vivienda a través de widgets intuitivos.

Predicción en Tiempo Real: Ofrece estimaciones de precio instantáneas basadas en las entradas del usuario.

Modelo Robusto: Utiliza un modelo XGBoost, conocido por su alto rendimiento y precisión.

Proceso Completo: Demuestra un ciclo de vida completo de un proyecto de Data Science, desde la limpieza de datos hasta el despliegue final.

🛠️ Tecnologías Utilizadas
Lenguaje: Python

Librerías de Análisis y Modelado: Pandas, NumPy, Scikit-learn, XGBoost

Aplicación Web: Streamlit

Control de Versiones y Despliegue: Git, GitHub, GitHub Codespaces, Streamlit Community Cloud






