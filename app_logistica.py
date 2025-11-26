import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc

st.set_page_config(page_title="Caso Logística – FastTrack", layout="wide")

st.title("Dashboard – Consumo de Combustible en FastTrack Logistics")

st.write("""
Este dashboard resume el análisis del caso de logística, donde se estudia el consumo
de combustible de una flota de camiones en función del tipo de ruta, el peso de carga
y la velocidad promedio.
""")


@st.cache_data
def cargar_datos():
    df = pd.read_csv("datos_logistica_transporte.csv", encoding="latin1")
    # Arreglo de posibles problemas de encoding
    df["Tipo_Ruta"] = df["Tipo_Ruta"].replace({"Monta¤a": "Montaña", "Montana": "Montaña"})
    return df

df = cargar_datos()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Exploración", "📈 ANOVA", "🔗 Correlación", "📘 Regresión", "🧮 Pronóstico y conclusiones"]
)

# ================= TAB 1: Exploración =================
with tab1:
    st.header("Exploración del dataset")

    st.subheader("Vista previa")
    st.dataframe(df.head())

    st.subheader("Estadísticas descriptivas")
    st.dataframe(df.describe().round(2))

    st.subheader("Distribución del consumo de combustible")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df["Consumo_Litros"], bins=15, kde=True, ax=ax)
    ax.set_xlabel("Consumo (L)")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

    st.subheader("Consumo por tipo de ruta")
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.boxplot(data=df, x="Tipo_Ruta", y="Consumo_Litros", ax=ax2)
    ax2.set_xlabel("Tipo de ruta")
    ax2.set_ylabel("Consumo (L)")
    st.pyplot(fig2)

# ================= TAB 2: ANOVA =================
with tab2:
    st.header("ANOVA: efecto del tipo de ruta")

    group_stats = df.groupby("Tipo_Ruta")[["Peso_Carga_Ton",
                                           "Velocidad_Promedio_kmh",
                                           "Consumo_Litros"]].agg(["mean", "std", "count"])
    st.subheader("Estadísticos por tipo de ruta")
    st.dataframe(group_stats.round(2))

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

# ================= TAB 3: Correlación =================
with tab3:
    st.header("Correlación entre variables numéricas")

    num_cols = ["Peso_Carga_Ton", "Velocidad_Promedio_kmh", "Consumo_Litros"]
    corr = df[num_cols].corr()

    st.subheader("Matriz de correlación (Pearson)")
    st.dataframe(corr.round(3))

    fig3, ax3 = plt.subplots(figsize=(5,4))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    st.write("""
Se observa una correlación positiva fuerte entre **Peso_Carga_Ton** y **Consumo_Litros**,
mientras que la relación entre velocidad y consumo es débil y negativa.
    """)

# ================= TAB 4: Regresión =================
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
    st.latex(
        r"\widehat{\text{Consumo\_Litros}} = "
        + f"{b0:.4f}"
        + r" + "
        + f"{b1:.4f}"
        + r"\,\text{Peso\_Carga\_Ton}"
        + r" "
        + f"{b2:.4f}"
        + r"\,\text{Velocidad\_Promedio\_kmh}"
    )

    st.subheader("Resumen del modelo (OLS)")
    st.text(modelo.summary())

    # Real vs predicho
    y_pred = modelo.predict(X)
    fig4, ax4 = plt.subplots(figsize=(5,5))
    ax4.scatter(y, y_pred, alpha=0.7)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax4.plot(lims, lims, "r--")
    ax4.set_xlabel("Consumo real (L)")
    ax4.set_ylabel("Consumo predicho (L)")
    ax4.set_title("Consumo real vs predicho")
    st.pyplot(fig4)

# ================= TAB 5: Pronóstico y conclusiones =================
with tab5:
    st.header("Pronóstico y conclusiones")

    st.subheader("Pronóstico de consumo")

    peso_input = st.number_input("Peso de carga (toneladas):", min_value=0.0, max_value=40.0, value=15.0, step=0.5)
    vel_input = st.number_input("Velocidad promedio (km/h):", min_value=30.0, max_value=120.0, value=80.0, step=5.0)

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

- El **peso de la carga** es el factor con mayor impacto sobre el consumo (correlación fuerte \
  y coeficiente de regresión grande), mientras que el efecto de la velocidad es más pequeño.

- El modelo de **regresión lineal múltiple** explica alrededor del 80% de la variabilidad \
  en el consumo, por lo que es una herramienta útil para estimar el uso de combustible \
  en función de peso y velocidad.

- Para reducir el consumo de diésel, la empresa debería priorizar la **optimización del peso \
  transportado** y evitar, en la medida de lo posible, el uso de rutas de Montaña para cargas pesadas.
    """)
