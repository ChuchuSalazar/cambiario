import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Análisis Econométrico TC USD", layout="wide")

st.title("📊 Análisis Econométrico del Tipo de Cambio USD")
st.markdown("**Análisis descriptivo e inferencial de la paridad cambiaria**")

# Cargar archivo
uploaded_file = st.file_uploader("Cargar archivo Excel (columnas: fecha, compra)", type=['xlsx', 'xls'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.lower().str.strip()
    
    # Renombrar columna si tiene "(bid)" o similares
    if 'compra (bid)' in df.columns:
        df.rename(columns={'compra (bid)': 'compra'}, inplace=True)
    
    # Parsear fechas con formatos mixtos (d/m/yy y d/m/yyyy)
    df['fecha'] = pd.to_datetime(df['fecha'], format='mixed', dayfirst=True)
    df = df.sort_values('fecha').reset_index(drop=True)
    
    # Variables temporales
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['trimestre'] = df['fecha'].dt.quarter
    df['semestre'] = df['fecha'].dt.month.apply(lambda x: 1 if x <= 6 else 2)
    df['primera_variacion'] = df['compra'].diff()
    df['log_compra'] = np.log(df['compra'])
    df['rendimiento'] = df['compra'].pct_change() * 100
    
    st.success(f"✅ Datos cargados: {len(df)} observaciones desde {df['fecha'].min().date()} hasta {df['fecha'].max().date()}")
    
    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Visualizaciones", "📊 Estadística Descriptiva", 
        "🔍 Análisis Inferencial", "📅 Comparaciones Temporales", "🎯 Tests Econométricos"
    ])
    
    # TAB 1: VISUALIZACIONES
    with tab1:
        st.header("Gráficos de Serie Temporal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Serie Original
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df['fecha'], y=df['compra'], 
                                     mode='lines', name='TC Compra',
                                     line=dict(color='#1f77b4', width=2)))
            fig1.update_layout(title='Serie Original - Tipo de Cambio',
                             xaxis_title='Fecha', yaxis_title='TC (unidades moneda local)',
                             hovermode='x unified', height=400)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Primera Variación
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df['fecha'], y=df['primera_variacion'], 
                                     mode='lines', name='Primera Diferencia',
                                     line=dict(color='#ff7f0e', width=1.5)))
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            fig2.update_layout(title='Primera Variación (Δ TC)',
                             xaxis_title='Fecha', yaxis_title='Cambio Absoluto',
                             hovermode='x unified', height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Logaritmo
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df['fecha'], y=df['log_compra'], 
                                     mode='lines', name='Log(TC)',
                                     line=dict(color='#2ca02c', width=2)))
            fig3.update_layout(title='Serie en Logaritmos',
                             xaxis_title='Fecha', yaxis_title='Log(TC)',
                             hovermode='x unified', height=400)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Rendimientos
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=df['fecha'], y=df['rendimiento'], 
                                     mode='lines', name='Rendimiento',
                                     line=dict(color='#d62728', width=1)))
            fig4.add_hline(y=0, line_dash="dash", line_color="gray")
            fig4.update_layout(title='Rendimientos Porcentuales (%)',
                             xaxis_title='Fecha', yaxis_title='Rendimiento (%)',
                             hovermode='x unified', height=400)
            st.plotly_chart(fig4, use_container_width=True)
        
        # Distribución
        st.subheader("Distribución de Valores")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig5 = go.Figure()
            fig5.add_trace(go.Histogram(x=df['compra'], nbinsx=30, 
                                       marker_color='#1f77b4', name='TC'))
            fig5.update_layout(title='Distribución TC Original', 
                             xaxis_title='Tipo de Cambio', yaxis_title='Frecuencia',
                             height=350)
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            fig6 = go.Figure()
            fig6.add_trace(go.Histogram(x=df['primera_variacion'].dropna(), nbinsx=30, 
                                       marker_color='#ff7f0e', name='Δ TC'))
            fig6.update_layout(title='Distribución Primera Variación', 
                             xaxis_title='Cambio', yaxis_title='Frecuencia',
                             height=350)
            st.plotly_chart(fig6, use_container_width=True)
        
        with col3:
            fig7 = go.Figure()
            fig7.add_trace(go.Histogram(x=df['rendimiento'].dropna(), nbinsx=30, 
                                       marker_color='#d62728', name='Rendimiento'))
            fig7.update_layout(title='Distribución Rendimientos', 
                             xaxis_title='Rendimiento (%)', yaxis_title='Frecuencia',
                             height=350)
            st.plotly_chart(fig7, use_container_width=True)
    
    # TAB 2: ESTADÍSTICA DESCRIPTIVA
    with tab2:
        st.header("Análisis Estadístico Descriptivo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📌 TC Original")
            stats_original = pd.DataFrame({
                'Métrica': ['Media', 'Mediana', 'Desv. Estándar', 'Varianza', 
                           'Mínimo', 'Máximo', 'Rango', 'Asimetría', 'Curtosis', 'CV (%)'],
                'Valor': [
                    df['compra'].mean(),
                    df['compra'].median(),
                    df['compra'].std(),
                    df['compra'].var(),
                    df['compra'].min(),
                    df['compra'].max(),
                    df['compra'].max() - df['compra'].min(),
                    df['compra'].skew(),
                    df['compra'].kurtosis(),
                    (df['compra'].std() / df['compra'].mean()) * 100
                ]
            })
            st.dataframe(stats_original.style.format({'Valor': '{:.4f}'}), 
                        use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("📌 Primera Variación")
            stats_var = pd.DataFrame({
                'Métrica': ['Media', 'Mediana', 'Desv. Estándar', 'Varianza', 
                           'Mínimo', 'Máximo', 'Rango', 'Asimetría', 'Curtosis'],
                'Valor': [
                    df['primera_variacion'].mean(),
                    df['primera_variacion'].median(),
                    df['primera_variacion'].std(),
                    df['primera_variacion'].var(),
                    df['primera_variacion'].min(),
                    df['primera_variacion'].max(),
                    df['primera_variacion'].max() - df['primera_variacion'].min(),
                    df['primera_variacion'].skew(),
                    df['primera_variacion'].kurtosis()
                ]
            })
            st.dataframe(stats_var.style.format({'Valor': '{:.4f}'}), 
                        use_container_width=True, hide_index=True)
        
        st.subheader("📌 Estadísticas por Percentiles")
        percentiles = df['compra'].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        perc_df = pd.DataFrame({
            'Percentil': ['1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '99%'],
            'Valor': percentiles.values
        })
        st.dataframe(perc_df.style.format({'Valor': '{:.4f}'}), 
                    use_container_width=True, hide_index=True)
    
    # TAB 3: ANÁLISIS INFERENCIAL
    with tab3:
        st.header("Análisis Inferencial y Tests de Hipótesis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔬 Test de Normalidad")
            
            # Jarque-Bera
            jb_stat, jb_pval = stats.jarque_bera(df['compra'].dropna())
            st.write("**Test Jarque-Bera (TC Original)**")
            st.write(f"Estadístico: {jb_stat:.4f}")
            st.write(f"P-valor: {jb_pval:.4f}")
            st.write(f"Resultado: {'✅ Normal' if jb_pval > 0.05 else '❌ No Normal'} (α=0.05)")
            
            st.write("---")
            
            # Shapiro-Wilk
            if len(df) <= 5000:
                sw_stat, sw_pval = stats.shapiro(df['compra'].dropna())
                st.write("**Test Shapiro-Wilk**")
                st.write(f"Estadístico: {sw_stat:.4f}")
                st.write(f"P-valor: {sw_pval:.4f}")
                st.write(f"Resultado: {'✅ Normal' if sw_pval > 0.05 else '❌ No Normal'} (α=0.05)")
        
        with col2:
            st.subheader("📊 Intervalos de Confianza")
            
            mean_tc = df['compra'].mean()
            std_tc = df['compra'].std()
            n = len(df['compra'])
            se = std_tc / np.sqrt(n)
            
            # IC 95%
            ci_95 = stats.t.interval(0.95, n-1, loc=mean_tc, scale=se)
            st.write("**IC 95% para la Media**")
            st.write(f"Límite Inferior: {ci_95[0]:.4f}")
            st.write(f"Media: {mean_tc:.4f}")
            st.write(f"Límite Superior: {ci_95[1]:.4f}")
            
            st.write("---")
            
            # IC 99%
            ci_99 = stats.t.interval(0.99, n-1, loc=mean_tc, scale=se)
            st.write("**IC 99% para la Media**")
            st.write(f"Límite Inferior: {ci_99[0]:.4f}")
            st.write(f"Media: {mean_tc:.4f}")
            st.write(f"Límite Superior: {ci_99[1]:.4f}")
        
        st.subheader("📈 Test de Tendencia")
        
        # Mann-Kendall simplificado
        n = len(df['compra'])
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(df['compra'].iloc[j] - df['compra'].iloc[i])
        
        var_s = n * (n-1) * (2*n+5) / 18
        z_mk = s / np.sqrt(var_s) if var_s > 0 else 0
        p_mk = 2 * (1 - stats.norm.cdf(abs(z_mk)))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Estadístico S", f"{s:,.0f}")
        col2.metric("Z-score", f"{z_mk:.4f}")
        col3.metric("P-valor", f"{p_mk:.4f}")
        
        if p_mk < 0.05:
            trend = "📈 Tendencia significativa" if s > 0 else "📉 Tendencia significativa"
            st.success(f"{trend} detectada (α=0.05)")
        else:
            st.info("➡️ No hay tendencia significativa (α=0.05)")
    
    # TAB 4: COMPARACIONES TEMPORALES
    with tab4:
        st.header("Análisis Comparativo por Períodos")
        
        # Por Año
        st.subheader("📅 Comparación Anual")
        annual = df.groupby('año')['compra'].agg([
            ('Promedio', 'mean'),
            ('Mediana', 'median'),
            ('Desv. Est.', 'std'),
            ('Mínimo', 'min'),
            ('Máximo', 'max'),
            ('Observaciones', 'count')
        ]).round(4)
        st.dataframe(annual, use_container_width=True)
        
        # Gráfico comparativo anual
        fig_annual = go.Figure()
        fig_annual.add_trace(go.Bar(x=annual.index, y=annual['Promedio'], 
                                   name='Promedio Anual',
                                   marker_color='lightblue'))
        fig_annual.update_layout(title='Tipo de Cambio Promedio por Año',
                                xaxis_title='Año', yaxis_title='TC Promedio',
                                height=400)
        st.plotly_chart(fig_annual, use_container_width=True)
        
        # Test ANOVA entre años
        años_unicos = df['año'].unique()
        if len(años_unicos) > 2:
            grupos_años = [df[df['año'] == año]['compra'].values for año in años_unicos]
            f_stat, p_val = stats.f_oneway(*grupos_años)
            st.write(f"**Test ANOVA entre años:** F={f_stat:.4f}, p-valor={p_val:.4f}")
            st.write(f"Resultado: {'✅ Diferencias significativas entre años' if p_val < 0.05 else '➡️ Sin diferencias significativas'} (α=0.05)")
        
        st.write("---")
        
        # Por Trimestre
        st.subheader("📊 Comparación Trimestral")
        df['año_trim'] = df['año'].astype(str) + '-Q' + df['trimestre'].astype(str)
        trimestral = df.groupby('año_trim')['compra'].agg([
            ('Promedio', 'mean'),
            ('Desv. Est.', 'std'),
            ('Mínimo', 'min'),
            ('Máximo', 'max')
        ]).round(4)
        st.dataframe(trimestral.tail(12), use_container_width=True)
        
        # Por Semestre
        st.subheader("📈 Comparación Semestral")
        df['año_sem'] = df['año'].astype(str) + '-S' + df['semestre'].astype(str)
        semestral = df.groupby('año_sem')['compra'].agg([
            ('Promedio', 'mean'),
            ('Desv. Est.', 'std'),
            ('Mínimo', 'min'),
            ('Máximo', 'max')
        ]).round(4)
        st.dataframe(semestral.tail(8), use_container_width=True)
        
        # Por Mes
        st.subheader("📆 Comparación Mensual (Estacionalidad)")
        mensual = df.groupby('mes')['compra'].agg([
            ('Promedio', 'mean'),
            ('Desv. Est.', 'std')
        ]).round(4)
        mensual.index = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                        'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        
        fig_mes = go.Figure()
        fig_mes.add_trace(go.Bar(x=mensual.index, y=mensual['Promedio'],
                                error_y=dict(type='data', array=mensual['Desv. Est.']),
                                marker_color='coral'))
        fig_mes.update_layout(title='TC Promedio por Mes (patrón estacional)',
                             xaxis_title='Mes', yaxis_title='TC Promedio',
                             height=400)
        st.plotly_chart(fig_mes, use_container_width=True)
    
    # TAB 5: TESTS ECONOMÉTRICOS
    with tab5:
        st.header("Tests Econométricos Avanzados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Test Dickey-Fuller Aumentado")
            st.write("**Prueba de Raíz Unitaria / Estacionariedad**")
            
            adf_result = adfuller(df['compra'].dropna(), autolag='AIC')
            
            st.write(f"Estadístico ADF: {adf_result[0]:.4f}")
            st.write(f"P-valor: {adf_result[1]:.4f}")
            st.write(f"Valores Críticos:")
            for key, value in adf_result[4].items():
                st.write(f"  - {key}: {value:.4f}")
            
            if adf_result[1] < 0.05:
                st.success("✅ Serie ESTACIONARIA (rechaza H0 de raíz unitaria)")
            else:
                st.warning("⚠️ Serie NO ESTACIONARIA (no rechaza H0)")
            
            st.write("---")
            
            # ADF en primera diferencia
            st.write("**ADF en Primera Diferencia**")
            adf_diff = adfuller(df['primera_variacion'].dropna(), autolag='AIC')
            st.write(f"Estadístico ADF: {adf_diff[0]:.4f}")
            st.write(f"P-valor: {adf_diff[1]:.4f}")
            
            if adf_diff[1] < 0.05:
                st.success("✅ Primera diferencia ESTACIONARIA")
            else:
                st.warning("⚠️ Primera diferencia NO ESTACIONARIA")
        
        with col2:
            st.subheader("📊 Autocorrelación")
            
            # ACF
            acf_values = acf(df['compra'].dropna(), nlags=20)
            
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Bar(x=list(range(len(acf_values))), 
                                    y=acf_values, marker_color='steelblue'))
            fig_acf.add_hline(y=1.96/np.sqrt(len(df)), line_dash="dash", 
                             line_color="red", annotation_text="IC 95%")
            fig_acf.add_hline(y=-1.96/np.sqrt(len(df)), line_dash="dash", 
                             line_color="red")
            fig_acf.update_layout(title='Función de Autocorrelación (ACF)',
                                 xaxis_title='Rezagos', yaxis_title='ACF',
                                 height=300)
            st.plotly_chart(fig_acf, use_container_width=True)
            
            # PACF
            pacf_values = pacf(df['compra'].dropna(), nlags=20)
            
            fig_pacf = go.Figure()
            fig_pacf.add_trace(go.Bar(x=list(range(len(pacf_values))), 
                                     y=pacf_values, marker_color='darkorange'))
            fig_pacf.add_hline(y=1.96/np.sqrt(len(df)), line_dash="dash", 
                              line_color="red", annotation_text="IC 95%")
            fig_pacf.add_hline(y=-1.96/np.sqrt(len(df)), line_dash="dash", 
                              line_color="red")
            fig_pacf.update_layout(title='Función de Autocorrelación Parcial (PACF)',
                                  xaxis_title='Rezagos', yaxis_title='PACF',
                                  height=300)
            st.plotly_chart(fig_pacf, use_container_width=True)
        
        st.subheader("📉 Volatilidad Histórica")
        
        window = st.slider("Ventana móvil (días)", 10, 120, 30)
        df['volatilidad'] = df['rendimiento'].rolling(window=window).std()
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=df['fecha'], y=df['volatilidad'],
                                    mode='lines', name='Volatilidad',
                                    line=dict(color='purple', width=2)))
        fig_vol.update_layout(title=f'Volatilidad Histórica (ventana {window} días)',
                             xaxis_title='Fecha', yaxis_title='Volatilidad (%)',
                             height=400)
        st.plotly_chart(fig_vol, use_container_width=True)
        
        st.write(f"**Volatilidad promedio:** {df['volatilidad'].mean():.4f}%")
        st.write(f"**Volatilidad máxima:** {df['volatilidad'].max():.4f}%")
        st.write(f"**Volatilidad mínima:** {df['volatilidad'].min():.4f}%")

else:
    st.info("👆 Por favor, carga un archivo Excel con las columnas 'fecha' y 'compra'")
    
    st.markdown("""
    ### 📋 Formato esperado del archivo:
    
    | fecha | compra |
    |-------|--------|
    | 2023-01-01 | 3.75 |
    | 2023-01-02 | 3.78 |
    | 2023-01-03 | 3.76 |
    
    **Características del análisis:**
    - ✅ Estadísticas descriptivas completas
    - ✅ Gráficos en nivel, primera variación y logaritmos
    - ✅ Tests de normalidad y estacionariedad
    - ✅ Comparaciones por año, trimestre, semestre y mes
    - ✅ Análisis de autocorrelación (ACF/PACF)
    - ✅ Tests de tendencia y volatilidad
    - ✅ Intervalos de confianza
    """)
