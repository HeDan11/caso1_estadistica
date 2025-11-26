import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc

# ==========================
# CONFIGURACIÓN GENERAL
# ==========================
st.set_page_config(page_title="Caso Logística – FastTrack", layout="wide")

st.title("Dashboard – Consumo de Combustible en FastTrack Logistics")

st.write("""
Este dashboard resume el análisis del caso de logística, donde se estudia el consumo
de combustible de una flota de camiones en función del tipo de ruta, el peso de carga
y la velocidad promedio.
""")

# ==========================
# CARGA DEL DATASET
# ==========================
@st.cache_data
def cargar_datos():
    df = pd.read_csv("datos_logistica_transporte.csv", encoding="latin1")
    # Ajuste por si viene mal codificado "Montaña"
    df["Tipo_Ruta"] = df["Tipo_Ruta"].replace({"Monta¤a": "Montaña", "Montana": "Montaña"})
    return df

df = cargar_datos()

# ==========================
# TABS
# ==========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Exploración", "📈 ANOVA", "🔗 Correlación", "📘 Regresión", "🧮 Pronóstico y conclusiones"]
)

# ================= TAB 1: EXPLORACIÓN =================
with tab1:
    st.header("Exploración del dataset")

    st.subheader("Vista previa")
    st.dataframe(df.head())

    st.subheader("Estadísticas descriptivas")
    st.dataframe(df.describe().round(2))

    st.subheader("Distribución del consumo de combustible")
    st.image("distribucion_precio.png", use_container_width=True)

# ================= TAB 2: ANOVA =================
with tab2:
    st.header("ANOVA: efecto del tipo de ruta")

    group_stats = df.groupby("Tipo_Ruta")[["Peso_Carga_Ton",
                                           "Velocidad_Promedio_kmh",
                                           "Consumo_Litros"]].agg(["mean", "std", "count"])
    st.subheader("Estadísticos por tipo de ruta")
    st.dataframe(group_stats.round(2))

    st.subheader("Consumo por tipo de ruta")
    st.image("boxplot_rutas.png", use_container_width=True)

    # ANOVA
    grupos = [g["Consumo_Litros"].values for _, g in df.groupby("Tipo_Ruta")]
    anova_res = stats.f_oneway(*grupos)

    st.subheader("Resultados del ANOVA de un factor")
    st.write(f"**F** = {anova_res.statistic:.2f}")
    st.write(f"**p-value** = {anova_res.pvalue:.3e}")

    st.write("""
Si el valor-p es menor que 0.05, se concluye que el tipo de ruta tiene un efecto
estadísticamente significativo sobre el consumo de combustible.
    """)

    # Tukey
    st.subheader("Prueba post-hoc de Tukey")
    comp = mc.MultiComparison(df["Consumo_Litros"], df["Tipo_Ruta"])
    tukey = comp.tukeyhsd()
    st.text(tukey.summary())

# ================= TAB 3: CORRELACIÓN =================
with tab3:
    st.header("Correlación entre variables numéricas")

    num_cols = ["Peso_Carga_Ton", "Velocidad_Promedio_kmh", "Consumo_Litros"]
    corr = df[num_cols].corr()

    st.subheader("Matriz de correlación (Pearson)")
    st.dataframe(corr.round(3))

    st.subheader("Heatmap de correlación")
    st.image("correlacion_viajes.png", use_container_width=True)

    st.write("""
Se observa una correlación positiva fuerte entre **Peso_Carga_Ton** y **Consumo_Litros**,
mientras que la relación entre velocidad y consumo es débil y negativa.
    """)

# ================= TAB 4: REGRESIÓN =================
with tab4:
    st.header("Modelo de regresión lineal múltiple")

    X = df[["Peso_Carga_Ton", "Velocidad_Promedio_kmh"]]
    X = sm.add_constant(X)
    y = df["Consumo_Litros"]

    modelo = sm.OLS(y, X).fit()

    b0 = modelo.params["const"]
    b1 = modelo.params["Peso_Carga_Ton"]
    b2 = modelo.params["Velocidad_Promedio_kmh"]

    st.subheader("Ecuación del modelo")

    # Construimos la ecuación en LaTeX con signos correctos
    eq = (
        r"\widehat{\text{Consumo\_Litros}} = "
        f"{b0:.2f} "
        f"{'+' if b1 >= 0 else '-'} {abs(b1):.2f}\,\text{{Peso\_Carga\_Ton}} "
        f"{'+' if b2 >= 0 else '-'} {abs(b2):.2f}\,\text{{Velocidad\_Promedio\_kmh}}"
    )
    st.latex(eq)

    st.subheader("Resumen del modelo (OLS)")
    st.text(modelo.summary())

    st.subheader("Consumo real vs predicho – Regresión múltiple")
    st.image("real_vs_predicho_consumo.png", use_container_width=True)

# ================= TAB 5: PRONÓSTICO Y CONCLUSIONES =================
with tab5:
    st.header("Pronóstico y conclusiones")

    st.subheader("Pronóstico de consumo")

    peso_input = st.number_input("Peso de carga (toneladas):", min_value=0.0, max_value=40.0,
                                 value=15.0, step=0.5)
    vel_input = st.number_input("Velocidad promedio (km/h):", min_value=30.0, max_value=120.0,
                                value=80.0, step=5.0)

    X_nuevo = pd.DataFrame({
        "const": [1],
        "Peso_Carga_Ton": [peso_input],
        "Velocidad_Promedio_kmh": [vel_input]
    })

    consumo_est = modelo.predict(X_nuevo)[0]
    st.write(f"**Consumo estimado:** {consumo_est:.2f} litros")

    st.markdown("---")
    st.subheader("Conclusiones principales")

    st.markdown("""
- El **tipo de ruta** influye de forma significativa en el consumo de combustible. \
  En particular, las rutas de **Montaña** presentan consumos medios mucho mayores \
  que las rutas Urbanas y de Autopista.

- El **peso de la carga** es el factor con mayor impacto sobre el consumo \
  (correlación fuerte y coeficiente de regresión grande), mientras que el efecto de la \
  velocidad es más pequeño.

- El modelo de **regresión lineal múltiple** explica alrededor del 80% de la variabilidad \
  en el consumo, por lo que es una herramienta útil para estimar el uso de combustible \
  en función de peso y velocidad.

- Para reducir el consumo de diésel, la empresa debería priorizar la **optimización del peso \
  transportado** y evitar, en la medida de lo posible, el uso de rutas de Montaña para cargas pesadas.
    """)

