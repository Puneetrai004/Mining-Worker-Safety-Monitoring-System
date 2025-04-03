import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import datetime
from data_generator import MiningDataGenerator
from model import MinerSafetyModel
from utils import (format_timestamp, get_risk_color, get_risk_icon, 
                  get_recommendations, create_vital_signs_gauge,
                  create_gas_level_gauge, create_time_series_chart)

# Set page configuration
st.set_page_config(
    page_title="Mining Worker Safety Monitoring",
    page_icon="⛑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #34495e;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .worker-card {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .normal-status {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 3px;
        text-align: center;
        font-weight: bold;
    }
    .alert-status {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 3px;
        text-align: center;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .alert-item {
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        border-radius: 3px;
    }
    .recommendation-item {
        background-color: #e2f0fb;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'worker_data' not in st.session_state:
    st.session_state.worker_data = {}
if 'worker_history' not in st.session_state:
    st.session_state.worker_history = {}
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'shift_start_time' not in st.session_state:
    st.session_state.shift_start_time = datetime.datetime.now()
if 'selected_worker' not in st.session_state:
    st.session_state.selected_worker = None

# Load or train model
@st.cache_resource
def load_model():
    if os.path.exists('model') and os.path.exists('data/scaler.pkl'):
        model = MinerSafetyModel()
        model.load('model', 'data/scaler.pkl')
        return model
    else:
        from train_model import train_and_save_model
        return train_and_save_model()

# Initialize data generator
@st.cache_resource
def get_data_generator():
    return MiningDataGenerator()

# Function to generate worker data
def generate_worker_data(worker_id, shift_hours, risk_bias=None):
    data_gen = get_data_generator()
    return data_gen.generate_real_time_data(worker_id, shift_hours, risk_bias)

# Function to predict risk
def predict_worker_risk(worker_data, model):
    if len(worker_data) < 10:
        return None
    
    # Get the last 10 records
    recent_data = worker_data.iloc[-10:].drop(['worker_id', 'timestamp', 'risk_class', 'risk_label'], axis=1)
    
    # Make prediction
    prediction = model.predict_risk(recent_data)
    return prediction

# Add a new alert to history
def add_alert(worker_id, worker_name, risk_type, probability, timestamp):
    st.session_state.alert_history.append({
        'worker_id': worker_id,
        'worker_name': worker_name,
        'risk_type': risk_type,
        'probability': probability,
        'timestamp': timestamp,
        'acknowledged': False
    })

# Calculate shift hours
def calculate_shift_hours():
    current_time = datetime.datetime.now()
    shift_duration = current_time - st.session_state.shift_start_time
    return shift_duration.total_seconds() / 3600  # Convert to hours

# Main app layout
def main():
    # Header
    st.markdown('<h1 class="main-header">⛑️ Mining Worker Safety Monitoring System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sub-header">Control Panel</h2>', unsafe_allow_html=True)
        
        # Add worker section
        st.subheader("Add Worker")
        worker_id = st.text_input("Worker ID", value="W" + str(np.random.randint(1000, 9999)))
        worker_name = st.text_input("Worker Name", value="John Doe")
        worker_role = st.selectbox("Role", ["Miner", "Supervisor", "Engineer", "Technician"])
        worker_age = st.slider("Age", 18, 65, 35)
        worker_experience = st.slider("Experience (years)", 0, 30, 5)
        
        if st.button("Add Worker"):
            if worker_id not in st.session_state.worker_data:
                st.session_state.worker_data[worker_id] = {
                    'name': worker_name,
                    'role': worker_role,
                    'age': worker_age,
                    'experience': worker_experience,
                    'active': True,
                    'last_updated': datetime.datetime.now(),
                    'current_risk': 0,
                    'risk_history': []
                }
                st.session_state.worker_history[worker_id] = pd.DataFrame()
                st.success(f"Worker {worker_name} added successfully!")
            else:
                st.error("Worker ID already exists!")
        
        # Monitoring controls
        st.markdown("---")
        st.subheader("Monitoring Controls")
        
        if st.button("Start Monitoring" if not st.session_state.monitoring_active else "Stop Monitoring"):
            st.session_state.monitoring_active = not st.session_state.monitoring_active
            if st.session_state.monitoring_active:
                st.session_state.shift_start_time = datetime.datetime.now()
        
        st.markdown(f"Status: {'🟢 Active' if st.session_state.monitoring_active else '🔴 Inactive'}")
        
        # Simulation controls
        st.markdown("---")
        st.subheader("Simulation Controls")
        
        sim_worker = st.selectbox("Select Worker", 
                                 options=["-"] + list(st.session_state.worker_data.keys()),
                                 format_func=lambda x: f"{st.session_state.worker_data[x]['name']} ({x})" if x != "-" else "Select worker")
        
        sim_risk = st.selectbox("Simulate Risk", 
                               options=["None", "Fatigue", "Gas Exposure", "Physical Stress", "Heat Stress"])
        
        risk_map = {"None": 0, "Fatigue": 1, "Gas Exposure": 2, "Physical Stress": 3, "Heat Stress": 4}
        
        if st.button("Trigger Simulation") and sim_worker != "-":
            # Generate data with risk bias
            risk_bias = risk_map[sim_risk]
            shift_hours = calculate_shift_hours()
            new_data = generate_worker_data(sim_worker, shift_hours, risk_bias)
            
            # Add to worker history
            if st.session_state.worker_history[sim_worker].empty:
                st.session_state.worker_history[sim_worker] = new_data
            else:
                st.session_state.worker_history[sim_worker] = pd.concat([st.session_state.worker_history[sim_worker], new_data])
            
            # Update worker data
            st.session_state.worker_data[sim_worker]['last_updated'] = datetime.datetime.now()
            st.session_state.worker_data[sim_worker]['current_risk'] = int(new_data['risk_class'].values[0])
