"""Streamlit web interface for Time Series Analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_preprocessing import TimeSeriesPreprocessor, generate_synthetic_data
from gru_model import GRUModel, GRUTrainer
from forecasting_methods import ForecastingMethods
from anomaly_detection import AnomalyDetector, AutoencoderAnomalyDetector
from visualization import TimeSeriesVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Time Series Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">📈 Time Series Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Data source selection
data_source = st.sidebar.selectbox(
    "Data Source",
    ["Real Stock Data", "Synthetic Data"],
    help="Choose between real stock data or synthetic data for testing"
)

if data_source == "Real Stock Data":
    symbol = st.sidebar.text_input("Stock Symbol", value="NFLX", help="Enter stock symbol (e.g., AAPL, GOOGL, MSFT)")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    column = st.sidebar.selectbox("Column", ["Close", "Open", "High", "Low", "Volume"])
else:
    n_samples = st.sidebar.slider("Number of Samples", 100, 2000, 1000)
    trend = st.sidebar.slider("Trend", 0.0, 0.1, 0.01, 0.001)
    seasonality_period = st.sidebar.slider("Seasonality Period", 7, 365, 30)
    noise_level = st.sidebar.slider("Noise Level", 0.01, 1.0, 0.1, 0.01)

# Model configuration
st.sidebar.header("Model Configuration")

# GRU parameters
gru_hidden_size = st.sidebar.slider("GRU Hidden Size", 10, 200, 50)
gru_layers = st.sidebar.slider("GRU Layers", 1, 5, 2)
gru_dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
sequence_length = st.sidebar.slider("Sequence Length", 5, 60, 30)
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
epochs = st.sidebar.slider("Epochs", 10, 200, 50)

# Forecasting methods
st.sidebar.header("Forecasting Methods")
use_arima = st.sidebar.checkbox("ARIMA", value=True)
use_auto_arima = st.sidebar.checkbox("Auto-ARIMA", value=True)
use_prophet = st.sidebar.checkbox("Prophet", value=True)
use_naive = st.sidebar.checkbox("Naive Methods", value=True)

# Anomaly detection methods
st.sidebar.header("Anomaly Detection")
use_isolation_forest = st.sidebar.checkbox("Isolation Forest", value=True)
use_statistical = st.sidebar.checkbox("Statistical Methods", value=True)
use_autoencoder = st.sidebar.checkbox("Autoencoder", value=False)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Load data button
if st.sidebar.button("Load Data", type="primary"):
    with st.spinner("Loading data..."):
        try:
            preprocessor = TimeSeriesPreprocessor()
            
            if data_source == "Real Stock Data":
                df = preprocessor.load_stock_data(symbol, str(start_date), str(end_date), column)
                data = df[column].values
                dates = df.index
            else:
                df = generate_synthetic_data(
                    n_samples=n_samples,
                    trend=trend,
                    seasonality_period=seasonality_period,
                    noise_level=noise_level
                )
                data = df['Close'].values
                dates = df.index
            
            st.session_state.data = data
            st.session_state.dates = dates
            st.session_state.preprocessor = preprocessor
            st.session_state.data_loaded = True
            
            st.success(f"Data loaded successfully! {len(data)} data points.")
            
        except Exception as e:
            st.error(f"Error loading data: {e}")

# Main content area
if st.session_state.data_loaded:
    data = st.session_state.data
    dates = st.session_state.dates
    preprocessor = st.session_state.preprocessor
    
    # Data overview
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Points", len(data))
    
    with col2:
        st.metric("Mean", f"{np.mean(data):.2f}")
    
    with col3:
        st.metric("Std Dev", f"{np.std(data):.2f}")
    
    with col4:
        st.metric("Min/Max", f"{np.min(data):.2f}/{np.max(data):.2f}")
    
    # Plot original data
    st.subheader("📈 Original Time Series")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=data,
        mode='lines',
        name='Data',
        line=dict(color='blue', width=1)
    ))
    
    fig.update_layout(
        title="Original Time Series Data",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Run analysis button
    if st.button("🚀 Run Complete Analysis", type="primary"):
        with st.spinner("Running analysis..."):
            try:
                # Prepare data
                scaled_data = preprocessor.normalize_data(data.reshape(-1, 1))
                X, y = preprocessor.create_sequences(scaled_data, sequence_length)
                X_train, X_test, y_train, y_test = preprocessor.train_test_split_sequences(X, y)
                
                # Initialize components
                forecaster = ForecastingMethods()
                anomaly_detector = AnomalyDetector()
                
                # Store results
                results = {}
                
                # GRU Model
                st.subheader("🧠 GRU Model Training")
                
                model = GRUModel(
                    input_size=1,
                    hidden_size=gru_hidden_size,
                    num_layers=gru_layers,
                    dropout=gru_dropout
                )
                
                trainer = GRUTrainer(model, learning_rate=learning_rate)
                
                # Create data loaders
                import torch
                from torch.utils.data import DataLoader, TensorDataset
                
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train),
                    torch.FloatTensor(y_train)
                )
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_test), torch.FloatTensor(y_test))
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                # Train model
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                history = trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    verbose=False
                )
                
                # Make predictions
                train_predictions = trainer.predict(train_loader)
                test_predictions = trainer.predict(val_loader)
                
                # Inverse transform
                train_pred_scaled = np.concatenate([X_train[:, -1, :], train_predictions])
                test_pred_scaled = np.concatenate([X_test[:, -1, :], test_predictions])
                
                train_pred_original = preprocessor.inverse_transform(train_pred_scaled)
                test_pred_original = preprocessor.inverse_transform(test_pred_scaled)
                
                results['GRU'] = {
                    'train_predictions': train_pred_original,
                    'test_predictions': test_pred_original,
                    'history': history
                }
                
                progress_bar.progress(25)
                status_text.text("GRU model training completed!")
                
                # Forecasting methods
                st.subheader("🔮 Forecasting Methods")
                
                forecast_results = {}
                
                if use_arima:
                    try:
                        result = forecaster.arima_forecast(data, forecast_steps=30)
                        forecast_results['ARIMA'] = result
                        st.success("ARIMA forecasting completed")
                    except Exception as e:
                        st.warning(f"ARIMA failed: {e}")
                
                if use_auto_arima:
                    try:
                        result = forecaster.auto_arima_forecast(data, forecast_steps=30)
                        forecast_results['Auto-ARIMA'] = result
                        st.success("Auto-ARIMA forecasting completed")
                    except Exception as e:
                        st.warning(f"Auto-ARIMA failed: {e}")
                
                if use_prophet:
                    try:
                        df_prophet = pd.DataFrame({'ds': dates, 'y': data})
                        result = forecaster.prophet_forecast(df_prophet, forecast_steps=30)
                        forecast_results['Prophet'] = result
                        st.success("Prophet forecasting completed")
                    except Exception as e:
                        st.warning(f"Prophet failed: {e}")
                
                if use_naive:
                    try:
                        result = forecaster.naive_forecast(data, forecast_steps=30, method='last')
                        forecast_results['Naive'] = result
                        st.success("Naive forecasting completed")
                    except Exception as e:
                        st.warning(f"Naive forecasting failed: {e}")
                
                results['forecasting'] = forecast_results
                progress_bar.progress(50)
                status_text.text("Forecasting methods completed!")
                
                # Anomaly detection
                st.subheader("🔍 Anomaly Detection")
                
                anomaly_results = {}
                
                if use_isolation_forest:
                    try:
                        result = anomaly_detector.isolation_forest_detection(data)
                        anomaly_results['Isolation Forest'] = result
                        st.success("Isolation Forest completed")
                    except Exception as e:
                        st.warning(f"Isolation Forest failed: {e}")
                
                if use_statistical:
                    try:
                        result = anomaly_detector.statistical_detection(data, method='zscore')
                        anomaly_results['Statistical Z-Score'] = result
                        st.success("Statistical Z-Score completed")
                    except Exception as e:
                        st.warning(f"Statistical detection failed: {e}")
                
                if use_autoencoder:
                    try:
                        detector = AutoencoderAnomalyDetector(input_dim=1)
                        detector.fit(data, epochs=50)
                        result = detector.detect_anomalies(data)
                        anomaly_results['Autoencoder'] = result
                        st.success("Autoencoder completed")
                    except Exception as e:
                        st.warning(f"Autoencoder failed: {e}")
                
                results['anomaly'] = anomaly_results
                progress_bar.progress(75)
                status_text.text("Anomaly detection completed!")
                
                # Store results in session state
                st.session_state.results = results
                st.session_state.analysis_complete = True
                
                progress_bar.progress(100)
                status_text.text("Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
    
    # Display results
    if st.session_state.analysis_complete:
        results = st.session_state.results
        
        # GRU Results
        if 'GRU' in results:
            st.header("🧠 GRU Model Results")
            
            gru_results = results['GRU']
            
            # Training history
            if 'history' in gru_results:
                history = gru_results['history']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history['train_losses'],
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='blue')
                ))
                
                if 'val_losses' in history and history['val_losses']:
                    fig.add_trace(go.Scatter(
                        y=history['val_losses'],
                        mode='lines',
                        name='Validation Loss',
                        line=dict(color='red')
                    ))
                
                fig.update_layout(
                    title="Training History",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Predictions
            train_pred = gru_results['train_predictions']
            test_pred = gru_results['test_predictions']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=data,
                mode='lines',
                name='Actual',
                line=dict(color='black', width=2)
            ))
            fig.add_trace(go.Scatter(
                y=train_pred,
                mode='lines',
                name='GRU Train',
                line=dict(color='blue', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                y=test_pred,
                mode='lines',
                name='GRU Test',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="GRU Model Predictions",
                xaxis_title="Time",
                yaxis_title="Value",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Forecasting Results
        if 'forecasting' in results and results['forecasting']:
            st.header("🔮 Forecasting Results")
            
            forecast_results = results['forecasting']
            
            # Create comparison plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=data,
                mode='lines',
                name='Actual',
                line=dict(color='black', width=2)
            ))
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, (method, result) in enumerate(forecast_results.items()):
                fig.add_trace(go.Scatter(
                    y=result['predictions'],
                    mode='lines',
                    name=f'{method} Forecast',
                    line=dict(color=colors[i % len(colors)], dash='dash')
                ))
            
            fig.update_layout(
                title="Forecasting Methods Comparison",
                xaxis_title="Time",
                yaxis_title="Value",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics comparison
            metrics_data = []
            for method, result in forecast_results.items():
                metrics = result['metrics']
                metrics_data.append({
                    'Method': method,
                    'MSE': metrics['mse'],
                    'MAE': metrics['mae'],
                    'R²': metrics['r2']
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.subheader("Forecasting Metrics")
            st.dataframe(metrics_df, use_container_width=True)
        
        # Anomaly Detection Results
        if 'anomaly' in results and results['anomaly']:
            st.header("🔍 Anomaly Detection Results")
            
            anomaly_results = results['anomaly']
            
            # Create anomaly plots
            for method, result in anomaly_results.items():
                st.subheader(f"{method} Detection")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=data,
                    mode='lines',
                    name='Data',
                    line=dict(color='blue')
                ))
                
                # Highlight anomalies
                anomalies = result['anomalies']
                anomaly_indices = np.where(anomalies)[0]
                
                if len(anomaly_indices) > 0:
                    fig.add_trace(go.Scatter(
                        x=anomaly_indices,
                        y=data[anomaly_indices],
                        mode='markers',
                        name=f'Anomalies ({len(anomaly_indices)})',
                        marker=dict(color='red', size=8)
                    ))
                
                fig.update_layout(
                    title=f"{method} Anomaly Detection",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Anomalies Detected", int(result['n_anomalies']))
                with col2:
                    st.metric("Anomaly Rate", f"{result['anomaly_rate']:.2%}")
                with col3:
                    if 'threshold' in result:
                        st.metric("Threshold", f"{result['threshold']:.3f}")

else:
    st.info("👈 Please load data using the sidebar to begin analysis.")
    
    # Show example
    st.subheader("📋 Example Usage")
    st.markdown("""
    1. **Configure Data Source**: Choose between real stock data or synthetic data
    2. **Set Parameters**: Adjust model parameters in the sidebar
    3. **Load Data**: Click "Load Data" to fetch and prepare the data
    4. **Run Analysis**: Click "Run Complete Analysis" to train models and generate results
    5. **Explore Results**: View forecasts, anomaly detection, and model performance
    
    ### Features:
    - **GRU Neural Network**: Advanced sequence modeling with configurable architecture
    - **Multiple Forecasting Methods**: ARIMA, Auto-ARIMA, Prophet, and Naive methods
    - **Anomaly Detection**: Isolation Forest, Statistical methods, and Autoencoder
    - **Interactive Visualizations**: Plotly-based charts with zoom and hover capabilities
    - **Real-time Metrics**: Performance comparison across different methods
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Time Series Analysis Dashboard | Built with Streamlit and PyTorch</p>
</div>
""", unsafe_allow_html=True)
