from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import io
import os

app = Flask(__name__)
CORS(app)

# variables globales
modelo = None
scaler = None
X_train = None
X_test = None
y_train = None
y_test = None
nombres_features = None
modelo_entrenado = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    datasets = [
        {
            'id': 'iris',
            'name': 'Iris Dataset',
            'description': 'Clasificacion de especies de flores iris (3 clases)',
            'samples': 150,
            'features': 4
        },
        {
            'id': 'wine',
            'name': 'Wine Dataset',
            'description': 'Clasificacion de cultivares de vino (3 clases)',
            'samples': 178,
            'features': 13
        },
        {
            'id': 'digits',
            'name': 'Digits Dataset',
            'description': 'Reconocimiento de digitos escritos a mano (10 clases)',
            'samples': 1797,
            'features': 64
        }
    ]
    return jsonify(datasets)


@app.route('/api/load-dataset/<dataset_id>', methods=['POST'])
def load_dataset(dataset_id):
    global X_train, X_test, y_train, y_test, nombres_features, modelo_entrenado
    
    try:
        if dataset_id == 'iris':
            from sklearn.datasets import load_iris
            data = load_iris()
            X = data.data
            y = data.target
            nombres_features = data.feature_names
        elif dataset_id == 'wine':
            from sklearn.datasets import load_wine
            data = load_wine()
            X = data.data
            y = data.target
            nombres_features = data.feature_names
        elif dataset_id == 'digits':
            from sklearn.datasets import load_digits
            data = load_digits()
            X = data.data
            y = data.target
            nombres_features = ['pixel_' + str(i) for i in range(X.shape[1])]
        else:
            return jsonify({'error': 'Dataset no encontrado'}), 400
        
        # separar train/test 80-20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        modelo_entrenado = False
        
        return jsonify({
            'success': True,
            'message': f'Dataset cargado: {len(X)} muestras',
            'data_info': {
                'samples': len(X),
                'features': X.shape[1],
                'feature_names': list(nombres_features),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'classes': int(len(np.unique(y)))
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    global X_train, X_test, y_train, y_test, nombres_features, modelo_entrenado
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporciono archivo'}), 400
        
        archivo = request.files['file']
        target_col = request.form.get('target_column', -1)
        
        # leer csv
        df = pd.read_csv(io.StringIO(archivo.stream.read().decode('utf-8')))
        
        # separar X y y
        if int(target_col) == -1:
            # ultima columna es target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            nombres_features = df.columns[:-1].tolist()
        else:
            target_col = int(target_col)
            X = df.drop(df.columns[target_col], axis=1).values
            y = df.iloc[:, target_col].values
            nombres_features = df.drop(df.columns[target_col], axis=1).columns.tolist()
        
        # separar train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        modelo_entrenado = False
        
        return jsonify({
            'success': True,
            'data_info': {
                'samples': len(X),
                'features': X.shape[1],
                'feature_names': nombres_features,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'classes': int(len(np.unique(y)))
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train_model():
    global modelo, scaler, X_train, X_test, y_train, y_test, modelo_entrenado
    
    try:
        if X_train is None:
            return jsonify({'error': 'No hay datos cargados'}), 400
        
        data = request.get_json()
        max_iter = data.get('max_iter', 1000)
        model_type = data.get('model_type', 'logistic')
        
        # normalizar datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # entrenar modelo segun tipo
        if model_type == 'linear':
            modelo = LinearRegression()
            modelo.fit(X_train_scaled, y_train)
            modelo_entrenado = True
            
            # evaluar en training
            y_pred_train = modelo.predict(X_train_scaled)
            train_mse = mean_squared_error(y_train, y_pred_train)
            
            return jsonify({
                'success': True,
                'message': 'Modelo lineal entrenado correctamente',
                'model_type': 'linear',
                'training_metrics': {
                    'mse': float(train_mse),
                    'rmse': float(np.sqrt(train_mse)),
                    'samples': len(X_train),
                    'features': X_train.shape[1]
                }
            })
        else:
            modelo = LogisticRegression(
                max_iter=max_iter,
                random_state=42,
                multi_class='multinomial'
            )
            modelo.fit(X_train_scaled, y_train)
            modelo_entrenado = True
            
            # evaluar en training
            y_pred_train = modelo.predict(X_train_scaled)
            train_accuracy = accuracy_score(y_train, y_pred_train)
            
            return jsonify({
                'success': True,
                'message': 'Modelo entrenado correctamente',
                'model_type': 'logistic',
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
    global modelo, scaler, X_test, y_test, modelo_entrenado
    
    try:
        if not modelo_entrenado or modelo is None:
            return jsonify({'error': 'Modelo no entrenado'}), 400
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = modelo.predict(X_test_scaled)
        
        # detectar si es regresion lineal
        es_lineal = isinstance(modelo, LinearRegression)
        
        if es_lineal:
            # metricas de regresion
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # si es clasificacion, redondear para calcular accuracy
            y_pred_rounded = np.round(y_pred).astype(int)
            y_test_int = y_test.astype(int) if hasattr(y_test, 'astype') else y_test
            acc = accuracy_score(y_test_int, y_pred_rounded) if len(np.unique(y_test)) < 20 else 0
            
            return jsonify({
                'success': True,
                'model_type': 'linear',
                'metrics': {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'accuracy': float(acc),
                    'test_samples': len(y_test)
                }
            })
        else:
            # metricas de clasificacion
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            return jsonify({
                'success': True,
                'model_type': 'logistic',
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
    global modelo, scaler, nombres_features, modelo_entrenado
    
    try:
        if not modelo_entrenado or modelo is None:
            return jsonify({'error': 'Modelo no entrenado'}), 400
        
        data = request.get_json()
        input_data = np.array(data['features']).reshape(1, -1)
        
        # normalizar y predecir
        input_scaled = scaler.transform(input_data)
        prediction = modelo.predict(input_scaled)[0]
        
        es_lineal = isinstance(modelo, LinearRegression)
        
        if es_lineal:
            return jsonify({
                'success': True,
                'prediction': float(round(prediction, 4)),
                'probabilities': None,
                'classes': None
            })
        else:
            probabilities = modelo.predict_proba(input_scaled)[0]
            return jsonify({
                'success': True,
                'prediction': int(prediction),
                'probabilities': probabilities.tolist(),
                'classes': modelo.classes_.tolist()
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    global modelo, nombres_features, modelo_entrenado
    
    try:
        if not modelo_entrenado or modelo is None:
            return jsonify({'error': 'Modelo no entrenado'}), 400
        
        # coeficientes del modelo
        coef = modelo.coef_
        intercept = modelo.intercept_
        
        es_lineal = isinstance(modelo, LinearRegression)
        if es_lineal:
            # LinearRegression devuelve coef_ 1D
            coef_list = coef.tolist() if hasattr(coef, 'tolist') else coef
            return jsonify({
                'success': True,
                'model_info': {
                    'trained': modelo_entrenado,
                    'classes': None,
                    'n_features': modelo.n_features_in_,
                    'feature_names': nombres_features,
                    'coefficients': coef_list if isinstance(coef_list, list) else [coef_list],
                    'intercept': [float(intercept)]
                }
            })
        else:
            return jsonify({
                'success': True,
                'model_info': {
                    'trained': modelo_entrenado,
                    'classes': modelo.classes_.tolist(),
                    'n_features': modelo.n_features_in_,
                    'feature_names': nombres_features,
                    'coefficients': coef.tolist(),
                    'intercept': intercept.tolist()
                }
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
