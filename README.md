#  Predicción de Riesgo Crediticio con Modelos de Ensamble

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-9CF?style=for-the-badge&logo=lightgbm&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Metodología CRISP-DM](https://img.shields.io/badge/Metodología-CRISP--DM-blue?style=for-the-badge)

Este repositorio contiene el desarrollo de un modelo predictivo para evaluar el riesgo de impago en créditos de consumo utilizando el dataset `credit_risk_dataset.csv`. El enfoque principal fue construir un pipeline que maneje datos del mundo real y comparase el rendimiento de algoritmos de Bagging y Boosting para la toma de decisiones financieras.

## Sobre el Conjunto de Datos (Dataset)

Este proyecto utiliza el **[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data)** disponible en Kaggle. 

El dataset modela el comportamiento histórico de una cartera de créditos de consumo y contiene **32,581 registros** con **12 variables** distribuidas entre información socioeconómica del solicitante, características financieras de la operación e historial crediticio:

* **Variable Objetivo:** `loan_status` (indicador de incumplimiento o impago).
* **Variables de Solicitante:** `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`.
* **Variables del Crédito:** `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`.
* **Historial Crediticio:** `cb_person_default_on_file`, `cb_person_cred_hist_length`.

##  Contexto y Aplicación Operativa
En operaciones de atención al cliente y gestión de cartera como entornos BPO/CRM, anticipar el impago permite:
* **Segmentar carteras de cobranza:** Priorizar esfuerzos de contacto y canales de comunicación en el contact center según el nivel de riesgo del cliente.
* **Automatizar aprobaciones:** Filtrar solicitudes con inconsistencias o perfiles de alto riesgo antes de pasar a una evaluación manual, reduciendo tiempos y costos operativos.

##  Hallazgos del Análisis Exploratorio y Saneamiento
Durante la auditoría visual y estadística de los datos se identificaron e intervinieron los siguientes problemas antes de entrenar los modelos:
* **Inconsistencias en registros:** Se detectaron y eliminaron valores atípicos imposibles, por ejemplo, edades de hasta 144 años o más de un siglo de experiencia laboral.
* **Imputación sin fuga de datos (*Data Leakage*):** Las variables faltantes (`person_emp_length` y `loan_int_rate`) se imputaron usando las medianas calculadas únicamente en el set de entrenamiento, heredándolas luego al set de prueba.
* **Codificación segura:** Se aplicó One-Hot Encoding a variables cualitativas como vivienda, intención del préstamo, etc. usando `drop_first=True` para evitar la multicolinealidad, sincronizando los sets de datos mediante alineación matricial (`.align`).

##  Modelado y Estrategia de Balanceo
El dataset presentaba un fuerte desbalance de clases (~78% de clientes cumplidos vs ~22% en default). Para evitar un modelo sesgado, se entrenaron dos algoritmos de ensamble configurando penalizaciones en sus funciones de pérdida:

1. **Random Forest (Bagging):** Configurado con `class_weight='balanced'`.
2. **LightGBM (Boosting):** Configurado con `scale_pos_weight=3.5`.

### Resultados en el Conjunto de Prueba:
| Modelo | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 92.10% | **88.13%** | 73.84% | **80.35%** |
| **LightGBM** | 91.32% | 81.27% | **78.35%** | 79.78% |

**Criterio de Selección:** Aunque Random Forest obtuvo un F1-Score ligeramente superior, **LightGBM fue seleccionado como el modelo ganador**. En riesgo crediticio, el costo financiero de un Falso Negativo, es decir, aprobar un crédito a alguien que no pagará es mayor que el de un Falso Positivo. LightGBM logró capturar mejor los casos de default reales (78.35% de Recall), minimizando de mejor manera las pérdidas potenciales de la operación.

El modelo final fue exportado a un archivo portable (`modelo_ganador_lightgbm.pkl`) usando `joblib` para quedar listo para su despliegue en producción.
