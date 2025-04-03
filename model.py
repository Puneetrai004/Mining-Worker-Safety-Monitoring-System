import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class MinerSafetyModel:
    def __init__(self, seq_length=10, num_classes=6):
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.model = None
        self.scaler = StandardScaler()
        self.risk_class_map = {
            0: 'Normal',
            1: 'Fatigue',
            2: 'Gas Exposure',
            3: 'Physical Stress',
            4: 'Heat Stress',
            5: 'Multiple Risks'
        }
    
    def prepare_sequences(self, data, labels, seq_length):
        """Prepare sequential data for LSTM model"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(labels[i + seq_length])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model for sequence classification"""
        model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X, y, epochs=20, batch_size=64, validation_split=0.2):
        """Train the model on the provided dataset"""
        if self.model is None:
            self.build_model((self.seq_length, X.shape[1]))
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Prepare sequences for LSTM
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train, self.seq_length)
        X_test_seq, y_test_seq = self.prepare_sequences(X_test, y_test, self.seq_length)
        
        # Convert labels to categorical for multi-class classification
        y_train_cat = to_categorical(y_train_seq, self.num_classes)
        y_test_cat = to_categorical(y_test_seq, self.num_classes)
        
        # Train the model
        history = self.model.fit(
            X_train_seq, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X_test_seq, y_test_cat)
        print(f"Test accuracy: {accuracy:.4f}")
        
        return history, (X_test_seq, y_test_cat)
    
    def predict_risk(self, new_data):
        """
        Predict risk from a sequence of sensor data
        
        Args:
            new_data: DataFrame with the same columns as training data, with seq_length rows
        
        Returns:
            Dictionary with risk class, type, and probability
        """
        # Scale the data
        new_data_scaled = self.scaler.transform(new_data)
        
        # Reshape for LSTM
        new_data_seq = new_data_scaled[-self.seq_length:].reshape(1, self.seq_length, -1)
        
        # Predict
        prediction = self.model.predict(new_data_seq)
        risk_class = np.argmax(prediction[0])
        risk_probability = prediction[0][risk_class]
        
        return {
            'risk_class': int(risk_class),
            'risk_type': self.risk_class_map[risk_class],
            'probability': float(risk_probability),
            'all_probabilities': prediction[0].tolist()
        }
    
    def save(self, model_path='model', scaler_path='data/scaler.pkl'):
        """Save the model and scaler"""
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
    
    def load(self, model_path='model', scaler_path='data/scaler.pkl'):
        """Load the model and scaler"""
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
