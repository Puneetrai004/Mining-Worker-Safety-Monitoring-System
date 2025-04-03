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
                      # Add alert if risk detected
            if int(new_data['risk_class'].values[0]) > 0:
                add_alert(
                    sim_worker, 
                    st.session_state.worker_data[sim_worker]['name'],
                    new_data['risk_label'].values[0], 
                    0.95,  # High probability for simulation
                    datetime.datetime.now()
                )
            
            st.success(f"Simulation triggered for {st.session_state.worker_data[sim_worker]['name']}!")
    
    # Main content
    if not st.session_state.worker_data:
        st.info("No workers added yet. Please add workers using the sidebar.")
    else:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Dashboard", "Worker Details", "Alert History"])
        
        with tab1:
            # Dashboard view
            st.markdown('<h2 class="sub-header">Real-time Monitoring Dashboard</h2>', unsafe_allow_html=True)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Workers", len(st.session_state.worker_data))
            with col2:
                active_workers = sum(1 for w in st.session_state.worker_data.values() if w['active'])
                st.metric("Active Workers", active_workers)
            with col3:
                workers_at_risk = sum(1 for w in st.session_state.worker_data.values() if w['current_risk'] > 0)
                st.metric("Workers at Risk", workers_at_risk)
            with col4:
                unacknowledged_alerts = sum(1 for a in st.session_state.alert_history if not a['acknowledged'])
                st.metric("Pending Alerts", unacknowledged_alerts)
            
            # Worker status cards
            st.markdown('<h3 class="sub-header">Worker Status</h3>', unsafe_allow_html=True)
            
            # Create a grid of worker cards
            cols = st.columns(3)
            for i, (worker_id, worker) in enumerate(st.session_state.worker_data.items()):
                with cols[i % 3]:
                    risk_color = get_risk_color(worker['current_risk'])
                    risk_icon = get_risk_icon(worker['current_risk'])
                    risk_label = "Normal" if worker['current_risk'] == 0 else next((a['risk_type'] for a in st.session_state.alert_history 
                                                                                if a['worker_id'] == worker_id and not a['acknowledged']), "Unknown")
                    
                    # Create worker card
                    st.markdown(f"""
                    <div class="worker-card" style="border-left: 5px solid {risk_color};">
                        <h4>{worker['name']} ({worker_id})</h4>
                        <p>Role: {worker['role']} | Age: {worker['age']} | Experience: {worker['experience']} years</p>
                        <div style="background-color: {risk_color}20; padding: 8px; border-radius: 4px;">
                            <span style="font-size: 1.5rem; margin-right: 8px;">{risk_icon}</span>
                            <span style="font-weight: bold; color: {risk_color};">{risk_label}</span>
                        </div>
                        <p style="font-size: 0.8rem; margin-top: 8px;">Last updated: {format_timestamp(worker['last_updated'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add button to view details
                    if st.button(f"View Details", key=f"view_{worker_id}"):
                        st.session_state.selected_worker = worker_id
                        # Switch to Worker Details tab
                        # Note: This doesn't work directly in Streamlit yet, but keeping for future compatibility
            
            # Recent alerts
            st.markdown('<h3 class="sub-header">Recent Alerts</h3>', unsafe_allow_html=True)
            
            if not st.session_state.alert_history:
                st.info("No alerts recorded yet.")
            else:
                # Show the 5 most recent alerts
                recent_alerts = sorted(st.session_state.alert_history, key=lambda x: x['timestamp'], reverse=True)[:5]
                
                for alert in recent_alerts:
                    risk_color = get_risk_color(next((k for k, v in MinerSafetyModel().risk_class_map.items() if v == alert['risk_type']), 0))
                    
                    st.markdown(f"""
                    <div class="alert-item" style="border-left: 5px solid {risk_color}; background-color: {risk_color}10;">
                        <strong>{alert['worker_name']} ({alert['worker_id']})</strong> - {alert['risk_type']}
                        <br>
                        <small>Detected at {format_timestamp(alert['timestamp'])}</small>
                        <span style="float: right; font-weight: bold; color: {risk_color};">
                            {int(alert['probability'] * 100)}% confidence
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            # Worker Details view
            st.markdown('<h2 class="sub-header">Worker Details</h2>', unsafe_allow_html=True)
            
            # Worker selection
            selected_worker = st.selectbox(
                "Select Worker",
                options=list(st.session_state.worker_data.keys()),
                format_func=lambda x: f"{st.session_state.worker_data[x]['name']} ({x})",
                key="worker_details_select"
            )
            
            if selected_worker:
                worker = st.session_state.worker_data[selected_worker]
                worker_history = st.session_state.worker_history.get(selected_worker, pd.DataFrame())
                
                # Worker info header
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <div style="font-size: 3rem; margin-right: 20px;">👷</div>
                    <div>
                        <h2 style="margin-bottom: 0;">{worker['name']}</h2>
                        <p style="margin-top: 0;">ID: {selected_worker} | Role: {worker['role']} | Age: {worker['age']} | Experience: {worker['experience']} years</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Current status
                risk_color = get_risk_color(worker['current_risk'])
                risk_label = "Normal" if worker['current_risk'] == 0 else next((a['risk_type'] for a in st.session_state.alert_history 
                                                                           if a['worker_id'] == selected_worker and not a['acknowledged']), "Unknown")
                
                st.markdown(f"""
                <div style="margin-bottom: 20px;">
                    <h3>Current Status</h3>
                    <div style="background-color: {risk_color}20; padding: 15px; border-radius: 8px; text-align: center;">
                        <span style="font-size: 2rem; margin-right: 10px;">{get_risk_icon(worker['current_risk'])}</span>
                        <span style="font-weight: bold; font-size: 1.5rem; color: {risk_color};">{risk_label}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # If we have data for this worker
                if not worker_history.empty:
                    # Get the most recent data point
                    latest_data = worker_history.iloc[-1]
                    
                    # Vital signs
                    st.markdown("<h3>Vital Signs</h3>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.plotly_chart(create_vital_signs_gauge(
                            latest_data['heart_rate'], 
                            "Heart Rate (bpm)", 
                            40, 140, 
                            [60, 100], 120
                        ))
                    
                    with col2:
                        st.plotly_chart(create_vital_signs_gauge(
                            latest_data['resp_rate'], 
                            "Respiration Rate (bpm)", 
                            5, 30, 
                            [12, 20], 25
                        ))
                    
                    with col3:
                        st.plotly_chart(create_vital_signs_gauge(
                            latest_data['body_temp'], 
                            "Body Temperature (°C)", 
                            35, 40, 
                            [36.5, 37.5], 38
                        ))
                    
                    # Environmental data
                    st.markdown("<h3>Environmental Conditions</h3>", unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.plotly_chart(create_vital_signs_gauge(
                            latest_data['env_temp'], 
                            "Temperature (°C)", 
                            10, 45, 
                            [18, 28], 35
                        ))
                    
                    with col2:
                        st.plotly_chart(create_vital_signs_gauge(
                            latest_data['humidity'], 
                            "Humidity (%)", 
                            0, 100, 
                            [30, 60], 80
                        ))
                    
                    with col3:
                        st.plotly_chart(create_gas_level_gauge(
                            latest_data['co_level'], 
                            "CO Level (ppm)", 
                            0, 10, 
                            2
                        ))
                    
                    with col4:
                        st.plotly_chart(create_gas_level_gauge(
                            latest_data['ch4_level'], 
                            "CH4 Level (%)", 
                            0, 5, 
                            1
                        ))
                    
                    # Time series data if we have enough points
                    if len(worker_history) > 1:
                        st.markdown("<h3>Historical Data</h3>", unsafe_allow_html=True)
                        
                        # Add timestamps as datetime for plotting
                        plot_data = worker_history.copy()
                        
                        # Heart rate over time
                        st.plotly_chart(create_time_series_chart(
                            plot_data, 'heart_rate', 'Heart Rate Over Time', 'red'
                        ))
                        
                        # Create two columns for additional charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.plotly_chart(create_time_series_chart(
                                plot_data, 'body_temp', 'Body Temperature Over Time', 'orange'
                            ))
                        
                        with col2:
                            st.plotly_chart(create_time_series_chart(
                                plot_data, 'env_temp', 'Environmental Temperature Over Time', 'blue'
                            ))
                else:
                    st.info("No sensor data available for this worker yet. Start monitoring or run a simulation to generate data.")
                
                # Recommendations if at risk
                if worker['current_risk'] > 0:
                    st.markdown("<h3>Safety Recommendations</h3>", unsafe_allow_html=True)
                    
                    recommendations = get_recommendations(risk_label)
                    for rec in recommendations:
                        st.markdown(f"""
                        <div class="recommendation-item">
                            <strong>→</strong> {rec}
                        </div>
                        """, unsafe_allow_html=True)
        
        with tab3:
            # Alert History view
            st.markdown('<h2 class="sub-header">Alert History</h2>', unsafe_allow_html=True)
            
            if not st.session_state.alert_history:
                st.info("No alerts recorded yet.")
            else:
                # Filter controls
                col1, col2 = st.columns(2)
                with col1:
                    filter_worker = st.selectbox(
                        "Filter by Worker",
                        options=["All"] + list(st.session_state.worker_data.keys()),
                        format_func=lambda x: "All Workers" if x == "All" else f"{st.session_state.worker_data[x]['name']} ({x})"
                    )
                
                with col2:
                    filter_risk = st.selectbox(
                        "Filter by Risk Type",
                        options=["All", "Fatigue", "Gas Exposure", "Physical Stress", "Heat Stress", "Multiple Risks"]
                    )
                
                # Apply filters
                filtered_alerts = st.session_state.alert_history
                if filter_worker != "All":
                    filtered_alerts = [a for a in filtered_alerts if a['worker_id'] == filter_worker]
                if filter_risk != "All":
                    filtered_alerts = [a for a in filtered_alerts if a['risk_type'] == filter_risk]
                
                # Sort by time (most recent first)
                filtered_alerts = sorted(filtered_alerts, key=lambda x: x['timestamp'], reverse=True)
                
                # Display alerts
                for i, alert in enumerate(filtered_alerts):
                    risk_color = get_risk_color(next((k for k, v in MinerSafetyModel().risk_class_map.items() if v == alert['risk_type']), 0))
                    
                    # Create columns for alert details and actions
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="alert-item" style="border-left: 5px solid {risk_color}; background-color: {risk_color}10;">
                            <strong>{alert['worker_name']} ({alert['worker_id']})</strong> - {alert['risk_type']}
                            <br>
                            <small>Detected at {alert['timestamp'].strftime("%Y-%m-%d %H:%M:%S")}</small>
                            <span style="float: right; font-weight: bold; color: {risk_color};">
                                {int(alert['probability'] * 100)}% confidence
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Acknowledge button
                        if not alert['acknowledged']:
                            if st.button("Acknowledge", key=f"ack_{i}"):
                                st.session_state.alert_history[st.session_state.alert_history.index(alert)]['acknowledged'] = True
                                st.success("Alert acknowledged!")
                        else:
                            st.markdown('<span style="color: green;">✓ Acknowledged</span>', unsafe_allow_html=True)
                    
                    # Show recommendations for this alert
                    with st.expander("View Recommendations"):
                        recommendations = get_recommendations(alert['risk_type'])
                        for rec in recommendations:
                            st.markdown(f"""
                            <div class="recommendation-item">
                                <strong>→</strong> {rec}
                            </div>
                            """, unsafe_allow_html=True)
    
    # Update data in real-time if monitoring is active
    if st.session_state.monitoring_active:
        # Calculate current shift hours
        shift_hours = calculate_shift_hours()
        
        # Update each active worker
        for worker_id, worker in st.session_state.worker_data.items():
            if worker['active']:
                # Generate new data point
                new_data = generate_worker_data(worker_id, shift_hours)
                
                # Add to worker history
                if st.session_state.worker_history[worker_id].empty:
                    st.session_state.worker_history[worker_id] = new_data
                else:
                    st.session_state.worker_history[worker_id] = pd.concat([st.session_state.worker_history[worker_id], new_data])
                
                # Update worker data
                st.session_state.worker_data[worker_id]['last_updated'] = datetime.datetime.now()
                st.session_state.worker_data[worker_id]['current_risk'] = int(new_data['risk_class'].values[0])
                
                # Add alert if risk detected
                if int(new_data['risk_class'].values[0]) > 0:
                    # Check if we already have an unacknowledged alert for this worker with this risk type
                    existing_alert = False
                    for alert in st.session_state.alert_history:
                        if (alert['worker_id'] == worker_id and 
                            alert['risk_type'] == new_data['risk_label'].values[0] and 
                            not alert['acknowledged']):
                            existing_alert = True
                            break
                    
                    if not existing_alert:
                        add_alert(
                            worker_id, 
                            worker['name'],
                            new_data['risk_label'].values[0], 
                            0.85 + np.random.random() * 0.1,  # Random high probability
                            datetime.datetime.now()
                        )
        
        # Rerun the app to update the UI
        time.sleep(2)  # Update every 2 seconds
        st.experimental_rerun()

if __name__ == "__main__":
    main()

