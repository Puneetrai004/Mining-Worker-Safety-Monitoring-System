import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_generator import MiningDataGenerator
from model import MinerSafetyModel

def train_and_save_model():
    # Create directories for model and data
    os.makedirs('model', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Generate dataset
    print("Generating dataset...")
    data_gen = MiningDataGenerator()
    df = data_gen.generate_dataset(20000)
    
    # Save dataset
    df.to_csv('data/mining_safety_dataset.csv', index=False)
    print("Dataset saved to data/mining_safety_dataset.csv")
    
    # Visualize the distribution of risk classes
    plt.figure(figsize=(10, 6))
    sns.countplot(x='risk_label', data=df)
    plt.title('Distribution of Risk Classes')
    plt.xlabel('Risk Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/risk_distribution.png')
    plt.close()
    
    # Prepare data for training
    X = df.drop(['risk_class', 'risk_label'], axis=1)
    y = df['risk_class']
    
    # Create and train model
    print("Training model...")
    safety_model = MinerSafetyModel()
    history, test_data = safety_model.train(X, y, epochs=10)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('data/training_history.png')
    plt.close()
    
    # Save model and scaler
    safety_model.save('model', 'data/scaler.pkl')
    print("Model and scaler saved")
    
    return safety_model

if __name__ == "__main__":
    train_and_save_model()
