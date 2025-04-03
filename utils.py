import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def format_timestamp(timestamp):
    """Format timestamp for display"""
    return timestamp.strftime("%H:%M:%S")

def get_risk_color(risk_class):
    """Get color for risk level"""
    colors = {
        0: "#00cc00",  # Green
        1: "#ffcc00",  # Yellow
        2: "#ff9900",  # Orange
        3: "#ff3300",  # Red
        4: "#cc00cc",  # Purple
        5: "#990000"   # Dark Red
    }
    return colors.get(risk_class, "#999999")  # Gray default

def get_risk_icon(risk_class):
    """Get icon for risk type"""
    icons = {
        0: "✓",       # Normal
        1: "😴",      # Fatigue
        2: "☁️",      # Gas
        3: "⚠️",      # Physical
        4: "🔥",      # Heat
        5: "⛔"       # Multiple
    }
    return icons.get(risk_class, "?")

def get_recommendations(risk_type):
    """Get recommendations based on risk type"""
    recommendations = {
        'Normal': "No action required. Continue monitoring.",
        'Fatigue': [
            "Worker should take a break immediately.",
            "Consider shift rotation or reduced hours.",
            "Check if worker has adequate rest between shifts.",
            "Provide caffeine or energy supplements if appropriate."
        ],
        'Gas Exposure': [
            "Evacuate area immediately.",
            "Check ventilation systems and gas monitors.",
            "Use respiratory protection if available.",
            "Identify and isolate the source of gas."
        ],
        'Physical Stress': [
            "Check equipment and posture.",
            "Reduce physical load or provide assistance.",
            "Ensure proper lifting techniques are being used.",
            "Assess for signs of musculoskeletal injury."
        ],
        'Heat Stress': [
            "Move to cooler area immediately.",
            "Provide hydration and cooling measures.",
            "Remove excess clothing/equipment if possible.",
            "Monitor for signs of heat exhaustion or heat stroke."
        ],
        'Multiple Risks': [
            "Immediate evacuation required.",
            "Multiple hazards detected - prioritize most severe.",
            "Call for emergency assistance if needed.",
            "Do not return until all hazards are addressed."
        ]
    }
    return recommendations.get(risk_type, ["Unknown risk type"])

def create_vital_signs_gauge(value, title, min_val, max_val, normal_range, danger_threshold):
    """Create a gauge chart for vital signs"""
    # Determine color based on value
    if min(normal_range) <= value <= max(normal_range):
        color = "green"
    elif value > danger_threshold:
        color = "red"
    else:
        color = "orange"
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': color},
            'steps': [
                {'range': [min_val, min(normal_range)], 'color': "lightgray"},
                {'range': normal_range, 'color': "lightgreen"},
                {'range': [max(normal_range), max_val], 'color': "lightgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': danger_threshold
            }
        }
    ))
    
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
    return fig

def create_gas_level_gauge(value, title, min_val, max_val, danger_threshold):
    """Create a gauge chart for gas levels"""
    # Determine color based on value
    if value < danger_threshold:
        color = "green"
    else:
        color = "red"
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': color},
            'steps': [
                {'range': [min_val, danger_threshold], 'color': "lightgreen"},
                {'range': [danger_threshold, max_val], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': danger_threshold
            }
        }
    ))
    
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
    return fig

def create_time_series_chart(data, y_column, title, color='blue'):
    """Create a time series chart for a specific metric"""
    fig = px.line(data, x='timestamp', y=y_column, title=title)
    fig.update_traces(line_color=color)
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig
