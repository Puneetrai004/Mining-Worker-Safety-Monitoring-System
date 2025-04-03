import numpy as np
import pandas as pd
import datetime

class MiningDataGenerator:
    def __init__(self, random_state=42):
        np.random.seed(random_state)
        self.risk_class_map = {
            0: 'Normal',
            1: 'Fatigue',
            2: 'Gas Exposure',
            3: 'Physical Stress',
            4: 'Heat Stress',
            5: 'Multiple Risks'
        }
    
    def generate_dataset(self, n_samples):
        """Generate a synthetic dataset of mining wearable sensor data"""
        # Time-related features
        shift_hours = np.random.uniform(0, 12, n_samples)
        time_of_day = np.random.uniform(0, 24, n_samples)
        
        # Worker physiological data
        heart_rate = np.zeros(n_samples)
        resp_rate = np.zeros(n_samples)
        body_temp = np.zeros(n_samples)
        
        # Motion data
        acc_x = np.random.normal(0, 1, n_samples)
        acc_y = np.random.normal(0, 1, n_samples)
        acc_z = np.random.normal(0, 1, n_samples)
        gyro_x = np.random.normal(0, 1, n_samples)
        gyro_y = np.random.normal(0, 1, n_samples)
        gyro_z = np.random.normal(0, 1, n_samples)
        
        # Environmental data
        env_temp = np.random.normal(25, 5, n_samples)
        humidity = np.random.uniform(30, 80, n_samples)
        co_level = np.random.exponential(0.5, n_samples)
        co2_level = np.random.normal(400, 50, n_samples)
        ch4_level = np.random.exponential(0.2, n_samples)
        o2_level = np.random.normal(20.9, 0.5, n_samples)
        h2s_level = np.random.exponential(0.1, n_samples)
        noise_level = np.random.normal(70, 10, n_samples)
        dust_level = np.random.exponential(1, n_samples)
        
        # Generate more realistic physiological data based on time and conditions
        for i in range(n_samples):
            # Heart rate increases with shift duration and environmental factors
            base_hr = np.random.normal(65, 5)
            hr_shift_factor = shift_hours[i] * 0.5  # Heart rate increases with shift duration
            hr_temp_factor = max(0, (env_temp[i] - 25) * 0.3)  # Heart rate increases in hot environments
            heart_rate[i] = base_hr + hr_shift_factor + hr_temp_factor + np.random.normal(0, 3)
            
            # Respiration rate correlates with heart rate and environmental factors
            base_rr = np.random.normal(14, 2)
            rr_hr_factor = (heart_rate[i] - 65) * 0.1
            rr_dust_factor = dust_level[i] * 0.5
            resp_rate[i] = base_rr + rr_hr_factor + rr_dust_factor + np.random.normal(0, 1)
            
            # Body temperature affected by environmental temperature and physical activity
            base_bt = np.random.normal(36.8, 0.1)
            bt_env_factor = max(0, (env_temp[i] - 25) * 0.03)
            bt_activity_factor = (heart_rate[i] - 65) * 0.01
            body_temp[i] = base_bt + bt_env_factor + bt_activity_factor + np.random.normal(0, 0.1)
        
        # Create the DataFrame
        data = {
            'shift_hours': shift_hours,
            'time_of_day': time_of_day,
            'heart_rate': heart_rate,
            'resp_rate': resp_rate,
            'body_temp': body_temp,
            'acc_x': acc_x,
            'acc_y': acc_y,
            'acc_z': acc_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z,
            'env_temp': env_temp,
            'humidity': humidity,
            'co_level': co_level,
            'co2_level': co2_level,
            'ch4_level': ch4_level,
            'o2_level': o2_level,
            'h2s_level': h2s_level,
            'noise_level': noise_level,
            'dust_level': dust_level
        }
        
        df = pd.DataFrame(data)
        
        # Calculate the magnitude of acceleration and gyroscope readings
        df['acc_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['gyro_magnitude'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
        
        # Generate risk labels based on conditions
        risk = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Class 1: Fatigue risk
            fatigue_risk = (df.loc[i, 'shift_hours'] > 8 and df.loc[i, 'heart_rate'] < 60) or \
                           (df.loc[i, 'heart_rate'] < 55) or \
                           (df.loc[i, 'shift_hours'] > 10)
            
            # Class 2: Harmful gas exposure
            gas_risk = (df.loc[i, 'co_level'] > 2) or \
                       (df.loc[i, 'co2_level'] > 1000) or \
                       (df.loc[i, 'ch4_level'] > 1) or \
                       (df.loc[i, 'o2_level'] < 19.5) or \
                       (df.loc[i, 'h2s_level'] > 1)
            
            # Class 3: Physical stress/injury risk
            physical_risk = (df.loc[i, 'acc_magnitude'] > 3) or \
                            (df.loc[i, 'gyro_magnitude'] > 3) or \
                            (df.loc[i, 'heart_rate'] > 100 and df.loc[i, 'resp_rate'] > 20)
            
            # Class 4: Heat stress
            heat_risk = (df.loc[i, 'env_temp'] > 30 and df.loc[i, 'humidity'] > 60) or \
                        (df.loc[i, 'body_temp'] > 37.8) or \
                        (df.loc[i, 'env_temp'] > 35)
            
            # Count how many risks are present
            risk_count = sum([fatigue_risk, gas_risk, physical_risk, heat_risk])
            
            if risk_count == 0:
                risk[i] = 0  # No risk
            elif risk_count >= 2:
                risk[i] = 5  # Multiple risks
            elif fatigue_risk:
                risk[i] = 1  # Fatigue risk
            elif gas_risk:
                risk[i] = 2  # Gas exposure risk
            elif physical_risk:
                risk[i] = 3  # Physical stress risk
            elif heat_risk:
                risk[i] = 4  # Heat stress risk
        
        df['risk_class'] = risk
        df['risk_label'] = df['risk_class'].map(self.risk_class_map)
        
        return df
    
    def generate_real_time_data(self, worker_id, shift_hours, risk_bias=None):
        """Generate a single data point for real-time monitoring"""
        # Create a single sample
        df = self.generate_dataset(1)
        
        # Override shift hours with the actual value
        df['shift_hours'] = shift_hours
        
        # If risk_bias is provided, manipulate the data to increase chance of that risk
        if risk_bias is not None and risk_bias > 0:
            if risk_bias == 1:  # Fatigue
                df['heart_rate'] = max(45, df['heart_rate'] - 15)
                df['shift_hours'] = min(12, df['shift_hours'] + 2)
            elif risk_bias == 2:  # Gas exposure
                df['co_level'] = df['co_level'] + 3
                df['o2_level'] = max(18, df['o2_level'] - 2)
            elif risk_bias == 3:  # Physical stress
                df['acc_magnitude'] = df['acc_magnitude'] + 3
                df['heart_rate'] = min(120, df['heart_rate'] + 30)
            elif risk_bias == 4:  # Heat stress
                df['env_temp'] = min(40, df['env_temp'] + 10)
                df['body_temp'] = min(38.5, df['body_temp'] + 1)
                df['humidity'] = min(90, df['humidity'] + 20)
        
        # Recalculate risk class based on modified data
        i = 0
        fatigue_risk = (df.loc[i, 'shift_hours'] > 8 and df.loc[i, 'heart_rate'] < 60) or \
                       (df.loc[i, 'heart_rate'] < 55) or \
                       (df.loc[i, 'shift_hours'] > 10)
        
        gas_risk = (df.loc[i, 'co_level'] > 2) or \
                   (df.loc[i, 'co2_level'] > 1000) or \
                   (df.loc[i, 'ch4_level'] > 1) or \
                   (df.loc[i, 'o2_level'] < 19.5) or \
                   (df.loc[i, 'h2s_level'] > 1)
        
        physical_risk = (df.loc[i, 'acc_magnitude'] > 3) or \
                        (df.loc[i, 'gyro_magnitude'] > 3) or \
                        (df.loc[i, 'heart_rate'] > 100 and df.loc[i, 'resp_rate'] > 20)
        
        heat_risk = (df.loc[i, 'env_temp'] > 30 and df.loc[i, 'humidity'] > 60) or \
                    (df.loc[i, 'body_temp'] > 37.8) or \
                    (df.loc[i, 'env_temp'] > 35)
        
        risk_count = sum([fatigue_risk, gas_risk, physical_risk, heat_risk])
        
        if risk_count == 0:
            risk_class = 0  # No risk
        elif risk_count >= 2:
            risk_class = 5  # Multiple risks
        elif fatigue_risk:
            risk_class = 1  # Fatigue risk
        elif gas_risk:
            risk_class = 2  # Gas exposure risk
        elif physical_risk:
            risk_class = 3  # Physical stress risk
        elif heat_risk:
            risk_class = 4  # Heat stress risk
        
        df['risk_class'] = risk_class
        df['risk_label'] = self.risk_class_map[risk_class]
        
        # Add worker ID and timestamp
        df['worker_id'] = worker_id
        df['timestamp'] = datetime.datetime.now()
        
        return df
