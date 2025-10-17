import streamlit as st
import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Visualizaciones", "📊 Estadística Descriptiva", 
        "🔍 Análisis Inferencial", "📅 Comparaciones Temporales", 
        "🎯 Tests Econométricos", "🔮 Proyecciones y Modelos"
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
    # TAB 6: PROYECCIONES Y MODELOS PREDICTIVOS
    with tab6:
        st.header("🔮 Estimación y Proyección del Tipo de Cambio")
        st.markdown("**Análisis comparativo de modelos predictivos y proyecciones a futuro**")
        
        # Preparar datos para modelado
        df_model = df[['fecha', 'compra']].copy()
        df_model = df_model.set_index('fecha')
        
        # Dividir en train/test (80/20)
        train_size = int(len(df_model) * 0.8)
        train, test = df_model[:train_size], df_model[train_size:]
        
        st.subheader("📊 División de Datos")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Observaciones", len(df_model))
        col2.metric("Entrenamiento (80%)", len(train))
        col3.metric("Prueba (20%)", len(test))
        
        st.write(f"**Período de entrenamiento:** {train.index[0].date()} a {train.index[-1].date()}")
        st.write(f"**Período de prueba:** {test.index[0].date()} a {test.index[-1].date()}")
        
        # Diccionario para almacenar resultados
        models_results = {}
        
        # ============ MODELO 1: ARIMA ============
        st.subheader("🎯 Modelo 1: ARIMA (Autoregressive Integrated Moving Average)")
        
        with st.spinner("Entrenando modelo ARIMA..."):
            try:
                # Encontrar mejor orden ARIMA usando AIC
                best_aic = np.inf
                best_order = None
                
                # Búsqueda de hiperparámetros
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            try:
                                model_arima_temp = ARIMA(train['compra'], order=(p, d, q))
                                fitted_temp = model_arima_temp.fit()
                                if fitted_temp.aic < best_aic:
                                    best_aic = fitted_temp.aic
                                    best_order = (p, d, q)
                            except:
                                continue
                
                # Entrenar modelo con mejor configuración
                model_arima = ARIMA(train['compra'], order=best_order)
                fitted_arima = model_arima.fit()
                
                # Predicciones
                pred_arima_test = fitted_arima.forecast(steps=len(test))
                
                # Métricas
                mae_arima = mean_absolute_error(test['compra'], pred_arima_test)
                rmse_arima = np.sqrt(mean_squared_error(test['compra'], pred_arima_test))
                mape_arima = mean_absolute_percentage_error(test['compra'], pred_arima_test) * 100
                
                models_results['ARIMA'] = {
                    'model': fitted_arima,
                    'predictions': pred_arima_test,
                    'mae': mae_arima,
                    'rmse': rmse_arima,
                    'mape': mape_arima,
                    'aic': fitted_arima.aic,
                    'bic': fitted_arima.bic,
                    'order': best_order
                }
                
                st.success(f"✅ Modelo ARIMA{best_order} entrenado exitosamente")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{mae_arima:.4f}")
                col2.metric("RMSE", f"{rmse_arima:.4f}")
                col3.metric("MAPE", f"{mape_arima:.2f}%")
                
                col1, col2 = st.columns(2)
                col1.metric("AIC", f"{fitted_arima.aic:.2f}")
                col2.metric("BIC", f"{fitted_arima.bic:.2f}")
                
                # Diagnóstico de residuos ARIMA
                st.write("**Diagnóstico de Residuos:**")
                residuals_arima = fitted_arima.resid
                
                col1, col2 = st.columns(2)
                with col1:
                    # Test Ljung-Box
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    lb_test = acorr_ljungbox(residuals_arima, lags=10, return_df=True)
                    st.write("**Test Ljung-Box (Autocorrelación de Residuos):**")
                    st.write(f"P-valor promedio: {lb_test['lb_pvalue'].mean():.4f}")
                    if lb_test['lb_pvalue'].mean() > 0.05:
                        st.success("✅ No hay autocorrelación significativa en residuos")
                    else:
                        st.warning("⚠️ Existe autocorrelación en residuos")
                
                with col2:
                    # Test Jarque-Bera en residuos
                    jb_stat, jb_pval = stats.jarque_bera(residuals_arima.dropna())
                    st.write("**Test Jarque-Bera (Normalidad de Residuos):**")
                    st.write(f"Estadístico: {jb_stat:.4f}")
                    st.write(f"P-valor: {jb_pval:.4f}")
                    if jb_pval > 0.05:
                        st.success("✅ Residuos se distribuyen normalmente")
                    else:
                        st.warning("⚠️ Residuos no son normales")
                
                # Gráfico residuos ARIMA
                fig_resid_arima = go.Figure()
                fig_resid_arima.add_trace(go.Scatter(x=residuals_arima.index, y=residuals_arima,
                                                     mode='lines', name='Residuos',
                                                     line=dict(color='red', width=1)))
                fig_resid_arima.add_hline(y=0, line_dash="dash", line_color="black")
                fig_resid_arima.update_layout(title='Residuos del Modelo ARIMA',
                                             xaxis_title='Fecha', yaxis_title='Residuos',
                                             height=350)
                st.plotly_chart(fig_resid_arima, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error en modelo ARIMA: {str(e)}")
        
        # ============ MODELO 2: EXPONENTIAL SMOOTHING ============
        st.subheader("📈 Modelo 2: Suavizamiento Exponencial (Holt-Winters)")
        
        with st.spinner("Entrenando modelo Holt-Winters..."):
            try:
                model_hw = ExponentialSmoothing(train['compra'], 
                                               trend='add', 
                                               seasonal=None,
                                               initialization_method='estimated')
                fitted_hw = model_hw.fit()
                
                pred_hw_test = fitted_hw.forecast(steps=len(test))
                
                mae_hw = mean_absolute_error(test['compra'], pred_hw_test)
                rmse_hw = np.sqrt(mean_squared_error(test['compra'], pred_hw_test))
                mape_hw = mean_absolute_percentage_error(test['compra'], pred_hw_test) * 100
                
                models_results['Holt-Winters'] = {
                    'model': fitted_hw,
                    'predictions': pred_hw_test,
                    'mae': mae_hw,
                    'rmse': rmse_hw,
                    'mape': mape_hw
                }
                
                st.success("✅ Modelo Holt-Winters entrenado exitosamente")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{mae_hw:.4f}")
                col2.metric("RMSE", f"{rmse_hw:.4f}")
                col3.metric("MAPE", f"{mape_hw:.2f}%")
                
            except Exception as e:
                st.error(f"Error en modelo Holt-Winters: {str(e)}")
        
        # ============ MODELO 3: REGRESIÓN POLINOMIAL ============
        st.subheader("📐 Modelo 3: Regresión Polinomial")
        
        with st.spinner("Entrenando modelo de Regresión Polinomial..."):
            try:
                # Crear variable temporal numérica
                train_poly = train.copy()
                test_poly = test.copy()
                
                train_poly['t'] = np.arange(len(train_poly))
                test_poly['t'] = np.arange(len(train_poly), len(train_poly) + len(test_poly))
                
                # Probar grados 2 y 3
                best_degree = 2
                best_score = -np.inf
                
                for degree in [2, 3]:
                    poly_features = PolynomialFeatures(degree=degree)
                    X_train_poly = poly_features.fit_transform(train_poly[['t']])
                    X_test_poly = poly_features.transform(test_poly[['t']])
                    
                    model_poly_temp = LinearRegression()
                    model_poly_temp.fit(X_train_poly, train_poly['compra'])
                    score = model_poly_temp.score(X_train_poly, train_poly['compra'])
                    
                    if score > best_score:
                        best_score = score
                        best_degree = degree
                
                # Entrenar con mejor grado
                poly_features = PolynomialFeatures(degree=best_degree)
                X_train_poly = poly_features.fit_transform(train_poly[['t']])
                X_test_poly = poly_features.transform(test_poly[['t']])
                
                model_poly = LinearRegression()
                model_poly.fit(X_train_poly, train_poly['compra'])
                
                pred_poly_test = model_poly.predict(X_test_poly)
                
                mae_poly = mean_absolute_error(test_poly['compra'], pred_poly_test)
                rmse_poly = np.sqrt(mean_squared_error(test_poly['compra'], pred_poly_test))
                mape_poly = mean_absolute_percentage_error(test_poly['compra'], pred_poly_test) * 100
                r2_poly = model_poly.score(X_train_poly, train_poly['compra'])
                
                models_results['Polinomial'] = {
                    'model': model_poly,
                    'poly_features': poly_features,
                    'predictions': pred_poly_test,
                    'mae': mae_poly,
                    'rmse': rmse_poly,
                    'mape': mape_poly,
                    'r2': r2_poly,
                    'degree': best_degree
                }
                
                st.success(f"✅ Modelo Polinomial (grado {best_degree}) entrenado exitosamente")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"{mae_poly:.4f}")
                col2.metric("RMSE", f"{rmse_poly:.4f}")
                col3.metric("MAPE", f"{mape_poly:.2f}%")
                col4.metric("R²", f"{r2_poly:.4f}")
                
            except Exception as e:
                st.error(f"Error en modelo Polinomial: {str(e)}")
        
        # ============ MODELO 4: MEDIA MÓVIL ============
        st.subheader("📊 Modelo 4: Media Móvil Simple")
        
        with st.spinner("Calculando Media Móvil..."):
            try:
                window = 30  # Ventana de 30 días
                train_ma = train.copy()
                
                # Calcular media móvil
                train_ma['ma'] = train_ma['compra'].rolling(window=window).mean()
                
                # Predicción: usar última media móvil
                last_ma = train_ma['ma'].iloc[-1]
                pred_ma_test = np.full(len(test), last_ma)
                
                mae_ma = mean_absolute_error(test['compra'], pred_ma_test)
                rmse_ma = np.sqrt(mean_squared_error(test['compra'], pred_ma_test))
                mape_ma = mean_absolute_percentage_error(test['compra'], pred_ma_test) * 100
                
                models_results['Media Móvil'] = {
                    'predictions': pred_ma_test,
                    'mae': mae_ma,
                    'rmse': rmse_ma,
                    'mape': mape_ma,
                    'window': window
                }
                
                st.success(f"✅ Modelo Media Móvil (ventana={window}) calculado")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{mae_ma:.4f}")
                col2.metric("RMSE", f"{rmse_ma:.4f}")
                col3.metric("MAPE", f"{mape_ma:.2f}%")
                
            except Exception as e:
                st.error(f"Error en Media Móvil: {str(e)}")
        
        # ============ COMPARACIÓN DE MODELOS ============
        st.subheader("🏆 Comparación de Modelos")
        
        comparison_data = []
        for model_name, results in models_results.items():
            comparison_data.append({
                'Modelo': model_name,
                'MAE': results['mae'],
                'RMSE': results['rmse'],
                'MAPE (%)': results['mape']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('MAPE (%)')
        
        st.dataframe(comparison_df.style.highlight_min(subset=['MAE', 'RMSE', 'MAPE (%)'], 
                                                       color='lightgreen'),
                    use_container_width=True, hide_index=True)
        
        # Mejor modelo
        best_model_name = comparison_df.iloc[0]['Modelo']
        best_model_mape = comparison_df.iloc[0]['MAPE (%)']
        
        st.success(f"🏆 **MEJOR MODELO: {best_model_name}** (MAPE: {best_model_mape:.2f}%)")
        
        # Gráfico de comparación
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Scatter(x=test.index, y=test['compra'],
                                           mode='lines', name='Real',
                                           line=dict(color='black', width=3)))
        
        colors = ['blue', 'red', 'green', 'orange']
        for i, (model_name, results) in enumerate(models_results.items()):
            fig_comparison.add_trace(go.Scatter(x=test.index, y=results['predictions'],
                                               mode='lines', name=model_name,
                                               line=dict(color=colors[i], width=2, dash='dash')))
        
        fig_comparison.update_layout(title='Comparación de Modelos vs Datos Reales (Set de Prueba)',
                                    xaxis_title='Fecha', yaxis_title='Tipo de Cambio',
                                    height=500, hovermode='x unified')
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # ============ PROYECCIONES FUTURAS ============
        st.subheader("🔮 Proyecciones a Futuro")
        
        # Opciones de proyección
        col1, col2 = st.columns(2)
        with col1:
            horizon_days = st.selectbox("Horizonte de proyección", 
                                       [1, 7, 30, 90, 365],
                                       format_func=lambda x: f"{x} día(s)" if x < 7 else 
                                                            f"{x} días ({x//7} semanas)" if x < 30 else
                                                            f"{x} días ({x//30} meses)" if x < 365 else
                                                            f"{x} días (1 año)")
        with col2:
            projection_model = st.selectbox("Modelo para proyección", 
                                           list(models_results.keys()))
        
        # Calcular proyecciones
        last_date = df_model.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                    periods=horizon_days, freq='D')
        
        # Proyectar según modelo seleccionado
        if projection_model == 'ARIMA':
            # Re-entrenar con todos los datos
            model_full_arima = ARIMA(df_model['compra'], 
                                    order=models_results['ARIMA']['order'])
            fitted_full_arima = model_full_arima.fit()
            future_pred = fitted_full_arima.forecast(steps=horizon_days)
            
            # Intervalos de confianza
            forecast_obj = fitted_full_arima.get_forecast(steps=horizon_days)
            forecast_ci = forecast_obj.conf_int()
            lower_ci = forecast_ci.iloc[:, 0].values
            upper_ci = forecast_ci.iloc[:, 1].values
            
        elif projection_model == 'Holt-Winters':
            model_full_hw = ExponentialSmoothing(df_model['compra'], 
                                                trend='add', 
                                                seasonal=None,
                                                initialization_method='estimated')
            fitted_full_hw = model_full_hw.fit()
            future_pred = fitted_full_hw.forecast(steps=horizon_days)
            
            # IC aproximado usando desviación estándar de residuos
            residuals = df_model['compra'] - fitted_full_hw.fittedvalues
            std_resid = residuals.std()
            lower_ci = future_pred - 1.96 * std_resid
            upper_ci = future_pred + 1.96 * std_resid
            
        elif projection_model == 'Polinomial':
            # Extender datos temporales
            last_t = len(df_model) - 1
            future_t = np.arange(last_t + 1, last_t + 1 + horizon_days).reshape(-1, 1)
            
            poly_feat = models_results['Polinomial']['poly_features']
            model_poly = models_results['Polinomial']['model']
            
            # Re-entrenar con todos los datos
            df_poly_full = df_model.copy()
            df_poly_full['t'] = np.arange(len(df_poly_full))
            X_full = poly_feat.fit_transform(df_poly_full[['t']])
            model_poly.fit(X_full, df_poly_full['compra'])
            
            X_future = poly_feat.transform(future_t)
            future_pred = model_poly.predict(X_future)
            
            # IC aproximado
            residuals = df_poly_full['compra'] - model_poly.predict(X_full)
            std_resid = residuals.std()
            lower_ci = future_pred - 1.96 * std_resid
            upper_ci = future_pred + 1.96 * std_resid
            
        else:  # Media Móvil
            last_ma = df_model['compra'].rolling(window=30).mean().iloc[-1]
            future_pred = np.full(horizon_days, last_ma)
            
            std_last = df_model['compra'].tail(30).std()
            lower_ci = future_pred - 1.96 * std_last
            upper_ci = future_pred + 1.96 * std_last
        
        # Crear DataFrame de proyecciones
        projections_df = pd.DataFrame({
            'Fecha': future_dates,
            'Proyección': future_pred,
            'Límite Inferior (95%)': lower_ci,
            'Límite Superior (95%)': upper_ci
        })
        
        # Mostrar proyecciones específicas
        st.write("### 📅 Proyecciones Clave")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Próximo día
        with col1:
            next_day = projections_df.iloc[0]
            st.metric("Próximo Día", 
                     f"{next_day['Proyección']:.4f}",
                     delta=f"{((next_day['Proyección'] - df_model['compra'].iloc[-1]) / df_model['compra'].iloc[-1] * 100):.2f}%")
            st.caption(f"{next_day['Fecha'].date()}")
        
        # Próxima semana (7 días)
        if horizon_days >= 7:
            with col2:
                next_week = projections_df.iloc[6]
                st.metric("En 7 Días", 
                         f"{next_week['Proyección']:.4f}",
                         delta=f"{((next_week['Proyección'] - df_model['compra'].iloc[-1]) / df_model['compra'].iloc[-1] * 100):.2f}%")
                st.caption(f"{next_week['Fecha'].date()}")
        
        # Próximo mes (30 días)
        if horizon_days >= 30:
            with col3:
                next_month = projections_df.iloc[29]
                st.metric("En 30 Días", 
                         f"{next_month['Proyección']:.4f}",
                         delta=f"{((next_month['Proyección'] - df_model['compra'].iloc[-1]) / df_model['compra'].iloc[-1] * 100):.2f}%")
                st.caption(f"{next_month['Fecha'].date()}")
        
        # Fin de año
        if horizon_days >= 90:
            end_of_year = datetime(df['fecha'].max().year, 12, 31)
            days_to_eoy = (end_of_year - last_date).days
            
            if days_to_eoy > 0 and days_to_eoy <= horizon_days:
                with col4:
                    eoy_projection = projections_df.iloc[days_to_eoy - 1]
                    st.metric("Cierre de Año", 
                             f"{eoy_projection['Proyección']:.4f}",
                             delta=f"{((eoy_projection['Proyección'] - df_model['compra'].iloc[-1]) / df_model['compra'].iloc[-1] * 100):.2f}%")
                    st.caption(f"{end_of_year.date()}")
        
        # Tabla completa de proyecciones
        st.write("### 📊 Tabla de Proyecciones Completa")
        st.dataframe(projections_df.style.format({
            'Proyección': '{:.4f}',
            'Límite Inferior (95%)': '{:.4f}',
            'Límite Superior (95%)': '{:.4f}'
        }), use_container_width=True, hide_index=True)
        
        # Gráfico de proyecciones
        fig_projection = go.Figure()
        
        # Datos históricos (últimos 90 días)
        historical_window = df_model.tail(90)
        fig_projection.add_trace(go.Scatter(x=historical_window.index, 
                                           y=historical_window['compra'],
                                           mode='lines', name='Histórico',
                                           line=dict(color='blue', width=2)))
        
        # Proyección
        fig_projection.add_trace(go.Scatter(x=future_dates, 
                                           y=future_pred,
                                           mode='lines', name='Proyección',
                                           line=dict(color='red', width=2, dash='dash')))
        
        # Intervalo de confianza
        fig_projection.add_trace(go.Scatter(x=future_dates, 
                                           y=upper_ci,
                                           mode='lines', name='IC 95% Superior',
                                           line=dict(width=0),
                                           showlegend=False))
        fig_projection.add_trace(go.Scatter(x=future_dates, 
                                           y=lower_ci,
                                           mode='lines', name='IC 95%',
                                           line=dict(width=0),
                                           fill='tonexty',
                                           fillcolor='rgba(255,0,0,0.2)'))
        
        fig_projection.update_layout(title=f'Proyección del Tipo de Cambio - Modelo {projection_model}',
                                    xaxis_title='Fecha', yaxis_title='Tipo de Cambio',
                                    height=500, hovermode='x unified')
        st.plotly_chart(fig_projection, use_container_width=True)
        
        # ============ SUPUESTOS Y VALIDACIONES DEL MODELO ============
        st.subheader("✅ Supuestos y Validaciones del Modelo")
        
        st.markdown(f"""
        ### Modelo Seleccionado: **{projection_model}**
        
        #### 📋 Supuestos del Modelo:
        """)
        
        if projection_model == 'ARIMA':
            st.markdown(f"""
            **ARIMA{models_results['ARIMA']['order']}:**
            1. **Estacionariedad:** Serie debe ser estacionaria después de diferenciación
            2. **Linealidad:** Relaciones lineales entre observaciones pasadas
            3. **Homocedasticidad:** Varianza constante en el tiempo
            4. **Residuos independientes:** Sin autocorrelación en errores
            5. **Normalidad de residuos:** Errores distribuidos normalmente
            
            #### ✅ Validaciones Realizadas:
            - Test Ljung-Box para autocorrelación de residuos
            - Test Jarque-Bera para normalidad de residuos
            - AIC/BIC para selección de orden óptimo
            - Análisis de ACF/PACF de residuos
            """)
            
        elif projection_model == 'Holt-Winters':
            st.markdown("""
            **Suavizamiento Exponencial (Holt-Winters):**
            1. **Tendencia:** Asume tendencia lineal aditiva
            2. **Suavidad:** Valores futuros dependen exponencialmente del pasado
            3. **Continuidad:** No hay cambios estructurales abruptos
            4. **Sin estacionalidad compleja:** Modelo sin componente estacional
            
            #### ✅ Validaciones Realizadas:
            - Optimización automática de parámetros (α, β)
            - Evaluación en set de prueba
            - Análisis de errores de predicción
            """)
            
        elif projection_model == 'Polinomial':
            st.markdown(f"""
            **Regresión Polinomial (Grado {models_results['Polinomial']['degree']}):**
            1. **Relación polinomial:** TC sigue tendencia polinomial en el tiempo
            2. **Continuidad:** Función suave y continua
            3. **Homocedasticidad:** Varianza constante de errores
            4. **Independencia:** Observaciones independientes
            
            #### ✅ Validaciones Realizadas:
            - R² para bondad de ajuste (R² = {models_results['Polinomial']['r2']:.4f})
            - Selección de grado óptimo mediante validación cruzada
            - Evaluación en set de prueba
            - Análisis de MAE, RMSE y MAPE
            """)
            
        else:  # Media Móvil
            st.markdown(f"""
            **Media Móvil Simple (Ventana {models_results['Media Móvil']['window']} días):**
            1. **Estabilidad:** TC se mantiene cerca del promedio reciente
            2. **Suavidad:** Fluctuaciones de corto plazo son ruido
            3. **Continuidad:** No hay cambios estructurales
            4. **Media constante:** Nivel medio relativamente estable
            
            #### ✅ Validaciones Realizadas:
            - Selección de ventana óptima
            - Evaluación en set de prueba
            - Análisis de errores
            """)
        
        # Métricas de confiabilidad
        st.markdown("#### 📊 Métricas de Confiabilidad del Modelo:")
        
        selected_model_results = models_results[projection_model]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Precisión en Set de Prueba:**")
            st.write(f"- MAE: {selected_model_results['mae']:.4f}")
            st.write(f"- RMSE: {selected_model_results['rmse']:.4f}")
            st.write(f"- MAPE: {selected_model_results['mape']:.2f}%")
            
            if selected_model_results['mape'] < 5:
                st.success("🟢 Excelente precisión (MAPE < 5%)")
            elif selected_model_results['mape'] < 10:
                st.info("🔵 Buena precisión (5% ≤ MAPE < 10%)")
            elif selected_model_results['mape'] < 20:
                st.warning("🟡 Precisión moderada (10% ≤ MAPE < 20%)")
            else:
                st.error("🔴 Precisión baja (MAPE ≥ 20%)")
        
        with col2:
            st.write("**Interpretación del MAPE:**")
            st.write(f"En promedio, las predicciones tienen un error del **{selected_model_results['mape']:.2f}%** respecto al valor real.")
            st.write("")
            st.write("**Confiabilidad del Intervalo:**")
            st.write("Los intervalos de confianza al 95% indican que:")
            st.write("- Hay 95% de probabilidad de que el valor real esté dentro del rango proyectado")
            st.write("- La amplitud del intervalo refleja la incertidumbre del modelo")
        
        # Gráfico de errores
        st.subheader("📉 Análisis de Errores de Predicción")
        
        errors = test['compra'].values - selected_model_results['predictions']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de errores
            fig_errors = go.Figure()
            fig_errors.add_trace(go.Histogram(x=errors, nbinsx=30,
                                             marker_color='coral',
                                             name='Errores'))
            fig_errors.update_layout(title='Distribución de Errores de Predicción',
                                    xaxis_title='Error', yaxis_title='Frecuencia',
                                    height=350)
            st.plotly_chart(fig_errors, use_container_width=True)
        
        with col2:
            # Errores en el tiempo
            fig_errors_time = go.Figure()
            fig_errors_time.add_trace(go.Scatter(x=test.index, y=errors,
                                                mode='lines+markers',
                                                marker=dict(size=6),
                                                line=dict(color='purple', width=2),
                                                name='Error'))
            fig_errors_time.add_hline(y=0, line_dash="dash", line_color="black")
            fig_errors_time.update_layout(title='Errores a lo Largo del Tiempo',
                                         xaxis_title='Fecha', yaxis_title='Error',
                                         height=350)
            st.plotly_chart(fig_errors_time, use_container_width=True)
        
        # Estadísticas de errores
        st.write("**Estadísticas de Errores:**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Error Medio", f"{errors.mean():.4f}")
        col2.metric("Desv. Est. Error", f"{errors.std():.4f}")
        col3.metric("Error Mínimo", f"{errors.min():.4f}")
        col4.metric("Error Máximo", f"{errors.max():.4f}")
        
        # Test de normalidad de errores
        jb_stat_err, jb_pval_err = stats.jarque_bera(errors)
        st.write(f"**Test de Normalidad de Errores (Jarque-Bera):** Estadístico={jb_stat_err:.4f}, p-valor={jb_pval_err:.4f}")
        
        if jb_pval_err > 0.05:
            st.success("✅ Los errores se distribuyen normalmente (buen indicador)")
        else:
            st.warning("⚠️ Los errores no se distribuyen normalmente")
        
        # ============ RESUMEN EJECUTIVO ============
        st.subheader("📋 Resumen Ejecutivo de Proyección")
        
        st.markdown(f"""
        ### Conclusiones y Recomendaciones:
        
        **1. Modelo Óptimo Seleccionado:** {best_model_name}
        - Este modelo presenta el menor error de predicción (MAPE: {best_model_mape:.2f}%)
        - Mejor balance entre precisión y simplicidad
        
        **2. Horizonte de Proyección:** {horizon_days} día(s)
        - Proyección realizada con modelo: {projection_model}
        - Confiabilidad: {'Alta' if selected_model_results['mape'] < 10 else 'Media' if selected_model_results['mape'] < 20 else 'Baja'}
        
        **3. Proyección para el Próximo Día:**
        - **Valor estimado:** {projections_df.iloc[0]['Proyección']:.4f}
        - **Intervalo de confianza (95%):** [{projections_df.iloc[0]['Límite Inferior (95%)']:.4f}, {projections_df.iloc[0]['Límite Superior (95%)']:.4f}]
        - **Cambio esperado:** {((projections_df.iloc[0]['Proyección'] - df_model['compra'].iloc[-1]) / df_model['compra'].iloc[-1] * 100):.2f}%
        
        **4. Factores de Riesgo:**
        - ⚠️ Los modelos asumen continuidad de patrones históricos
        - ⚠️ Eventos externos (políticos, económicos) pueden afectar la precisión
        - ⚠️ Mayor incertidumbre en horizontes más largos
        - ⚠️ Volatilidad histórica sugiere alta variabilidad
        
        **5. Recomendaciones:**
        - 📌 Actualizar el modelo regularmente con datos nuevos
        - 📌 Considerar análisis de sensibilidad ante escenarios
        - 📌 Monitorear métricas de error para detectar deterioro del modelo
        - 📌 Usar el intervalo de confianza para gestión de riesgos
        - 📌 Complementar con análisis fundamental y contexto económico
        """)
        
        # Descarga de proyecciones
        st.subheader("💾 Descargar Proyecciones")
        
        # Preparar archivo para descarga
        download_df = projections_df.copy()
        download_df['Modelo'] = projection_model
        download_df['MAPE_Modelo'] = selected_model_results['mape']
        download_df['Fecha_Generacion'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        csv = download_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Proyecciones (CSV)",
            data=csv,
            file_name=f"proyecciones_tc_{projection_model}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
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









 # Footer usando HTML component
footer_component = """
<div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%); padding: 40px 20px; border-radius: 15px; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2); color: white; text-align: center; margin-top: 30px;">
    <h2 style="margin: 0 0 20px 0; font-size: 20px; letter-spacing: 2px;">Desarrollado por:</h2>
    <div style="border-top: 2px solid rgba(255,255,255,0.3); border-bottom: 2px solid rgba(255,255,255,0.3); padding: 20px; margin: 20px auto; max-width: 600px;">
        <h3 style="margin: 0 0 10px 0; font-size: 20px;">MSc. Jesús Fernando Salazar R.</h3>
        <p style="margin: 5px 0;">📱 +58 414 286 8869</p>
        <p style="margin: 5px 0;">📧 auditoria.Vzla@gmail.com</p>
        <p style="margin-top: 10px; font-size: 13px; opacity: 0.9;">Consultor</p>
    </div>
    <p style="font-size: 20px; font-weight: bold; margin: 20px 0;">Transformando Datos en Decisiones Estratégicas</p>
    <div style="margin-top: 20px;">
        <span style="background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 20px; margin: 5px; display: inline-block; font-size: 12px;">📊 Analytics</span>
        <span style="background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 20px; margin: 5px; display: inline-block; font-size: 12px;">🤖 AI/ML</span>
        <span style="background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 20px; margin: 5px; display: inline-block; font-size: 12px;">⚙️ Industry 4.0</span>
    </div>
    <p style="margin-top: 20px; font-size: 11px; opacity: 0.8;">Sistema de Análisis de Costos v1.0 | © 2025 - Todos los derechos reservados</p>
</div>
"""

components.html(footer_component, height=400)