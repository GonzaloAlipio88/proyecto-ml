from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import io
import os

app = Flask(__name__)
CORS(app)

# Variables globales para almacenar el modelo y datos
model = None
scaler = None
X_train = None
X_test = None
y_train = None
y_test = None
feature_names = None
model_trained = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Retorna lista de datasets disponibles"""
    datasets = [
        {
            'id': 'iris',
            'name': 'Iris Dataset',
            'description': 'Clasificación de especies de flores iris (3 clases)',
            'samples': 150,
            'features': 4
        },
        {
            'id': 'wine',
            'name': 'Wine Dataset',
            'description': 'Clasificación de cultivares de vino (3 clases)',
            'samples': 178,
            'features': 13
        },
        {
            'id': 'digits',
            'name': 'Digits Dataset',
            'description': 'Reconocimiento de dígitos escritos a mano (10 clases)',
            'samples': 1797,
            'features': 64
        }
    ]
    return jsonify(datasets)

@app.route('/api/load-dataset/<dataset_id>', methods=['POST'])
def load_dataset(dataset_id):
    """Carga un dataset de ejemplo"""
    global X_train, X_test, y_train, y_test, feature_names, model_trained
    
    try:
        if dataset_id == 'iris':
            from sklearn.datasets import load_iris
            data = load_iris()
            X = data.data
            y = data.target
            feature_names = data.feature_names
        elif dataset_id == 'wine':
            from sklearn.datasets import load_wine
            data = load_wine()
            X = data.data
            y = data.target
            feature_names = data.feature_names
        elif dataset_id == 'digits':
            from sklearn.datasets import load_digits
            data = load_digits()
            X = data.data
            y = data.target
            feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
        else:
            return jsonify({'error': 'Dataset no encontrado'}), 400
        
        # Separar en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model_trained = False
        
        return jsonify({
            'success': True,
            'message': f'Dataset cargado: {len(X)} muestras',
            'data_info': {
                'samples': len(X),
                'features': X.shape[1],
                'feature_names': list(feature_names),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'classes': int(len(np.unique(y)))
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    """Carga datos desde un archivo CSV"""
    global X_train, X_test, y_train, y_test, feature_names, model_trained
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó archivo'}), 400
        
        file = request.files['file']
        target_col = request.form.get('target_column', -1)
        
        # Leer CSV
        df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
        
        # Separar X e y
        if int(target_col) == -1:
            # Última columna es target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            feature_names = df.columns[:-1].tolist()
        else:
            target_col = int(target_col)
            X = df.drop(df.columns[target_col], axis=1).values
            y = df.iloc[:, target_col].values
            feature_names = df.drop(df.columns[target_col], axis=1).columns.tolist()
        
        # Separar train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model_trained = False
        
        return jsonify({
            'success': True,
            'data_info': {
                'samples': len(X),
                'features': X.shape[1],
                'feature_names': feature_names,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'classes': int(len(np.unique(y)))
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Entrena el modelo de regresión logística"""
    global model, scaler, X_train, X_test, y_train, y_test, model_trained
    
    try:
        if X_train is None:
            return jsonify({'error': 'No hay datos cargados'}), 400
        
        data = request.get_json()
        max_iter = data.get('max_iter', 1000)
        
        # Normalizar datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Entrenar modelo
        model = LogisticRegression(
            max_iter=max_iter,
            random_state=42,
            multi_class='multinomial'
        )
        model.fit(X_train_scaled, y_train)
        model_trained = True
        
        # Evaluar en training
        y_pred_train = model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        
        return jsonify({
            'success': True,
            'message': 'Modelo entrenado correctamente',
            'training_metrics': {
                'accuracy': float(train_accuracy),
                'samples': len(X_train),
                'features': X_train.shape[1]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate', methods=['GET'])
def evaluate_model():
    """Evalúa el modelo en test set"""
    global model, scaler, X_test, y_test, model_trained
    
    try:
        if not model_trained or model is None:
            return jsonify({'error': 'Modelo no entrenado'}), 400
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        # Reporte de clasificación
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        return jsonify({
            'success': True,
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'test_samples': len(y_test)
            },
            'confusion_matrix': cm.tolist(),
            'classes': [int(c) for c in np.unique(y_test)],
            'report': report
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Realiza predicción en nuevos datos"""
    global model, scaler, feature_names, model_trained
    
    try:
        if not model_trained or model is None:
            return jsonify({'error': 'Modelo no entrenado'}), 400
        
        data = request.get_json()
        input_data = np.array(data['features']).reshape(1, -1)
        
        # Normalizar y predecir
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probabilities': probabilities.tolist(),
            'classes': model.classes_.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Retorna información del modelo entrenado"""
    global model, feature_names, model_trained
    
    try:
        if not model_trained or model is None:
            return jsonify({'error': 'Modelo no entrenado'}), 400
        
        # Coeficientes del modelo
        coef = model.coef_
        intercept = model.intercept_
        
        return jsonify({
            'success': True,
            'model_info': {
                'trained': model_trained,
                'classes': model.classes_.tolist(),
                'n_features': model.n_features_in_,
                'feature_names': feature_names,
                'coefficients': coef.tolist(),
                'intercept': intercept.tolist()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
