// Estado
let appState = {
    dataLoaded: false,
    modelTrained: false,
    featureNames: [],
    featureCount: 0
};

// Funciones auxiliares
const showLoading = () => document.getElementById('loading').style.display = 'flex';
const hideLoading = () => document.getElementById('loading').style.display = 'none';

const showError = (message) => {
    const errorBox = document.getElementById('errorBox');
    document.getElementById('errorMessage').textContent = message;
    errorBox.style.display = 'block';
    setTimeout(() => errorBox.style.display = 'none', 5000);
};

const showSuccess = (message) => {
    console.log(message);
};

const API_BASE = '/api';

// Cargar dataset de ejemplo
async function loadDataset(datasetId) {
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/load-dataset/${datasetId}`, {
            method: 'POST'
        });
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error);
        
        appState.dataLoaded = true;
        appState.featureNames = data.data_info.feature_names;
        appState.featureCount = data.data_info.features;
        
        updateDataInfo(data.data_info);
        showSuccess(`Dataset ${datasetId} cargado`);
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// Subir CSV
async function uploadCSV() {
    const file = document.getElementById('csvFile').files[0];
    if (!file) {
        showError('Selecciona un archivo CSV');
        return;
    }
    
    showLoading();
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('target_column', document.getElementById('targetCol').value);
        
        const response = await fetch(`${API_BASE}/upload-csv`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error);
        
        appState.dataLoaded = true;
        appState.featureNames = data.data_info.feature_names;
        appState.featureCount = data.data_info.features;
        
        updateDataInfo(data.data_info);
        showSuccess('CSV cargado correctamente');
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// Actualizar información de datos
function updateDataInfo(info) {
    document.getElementById('dataInfo').style.display = 'block';
    document.getElementById('samplesCount').textContent = info.samples;
    document.getElementById('featuresCount').textContent = info.features;
    document.getElementById('trainTestSplit').textContent = `${info.train_size} / ${info.test_size}`;
    document.getElementById('classesCount').textContent = info.classes;
}

// Entrenar modelo
async function trainModel() {
    if (!appState.dataLoaded) {
        showError('Carga datos primero');
        return;
    }
    
    showLoading();
    try {
        const maxIter = document.getElementById('maxIter').value;
        const response = await fetch(`${API_BASE}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ max_iter: parseInt(maxIter) })
        });
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error);
        
        appState.modelTrained = true;
        
        document.getElementById('trainStatus').style.display = 'block';
        document.getElementById('trainMessage').textContent = 
            `Modelo entrenado. Accuracy en train: ${(data.training_metrics.accuracy * 100).toFixed(2)}%`;
        
        // Cargar info del modelo
        await loadModelInfo();
        
        // Mostrar formulario de predicción
        showPredictionForm();
        
        showSuccess('Modelo entrenado exitosamente');
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// Cargar info del modelo
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/model-info`);
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error);
        
        const info = data.model_info;
        document.getElementById('modelInfo').style.display = 'block';
        document.getElementById('noModel').style.display = 'none';
        document.getElementById('modelClasses').textContent = info.classes.join(', ');
        document.getElementById('modelFeatures').textContent = info.n_features;
        
        // Mostrar coeficientes
        displayCoefficients(info.feature_names, info.coefficients);
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

// Mostrar coeficientes
function displayCoefficients(featureNames, coefficients) {
    const display = document.getElementById('coefficientsDisplay');
    display.innerHTML = '<strong>Coeficientes del Modelo:</strong>';
    
    coefficients.forEach((coef, idx) => {
        const feature = featureNames[idx] || `Característica ${idx}`;
        const values = Array.isArray(coef) ? coef : [coef];
        
        values.forEach((val, i) => {
            const item = document.createElement('div');
            item.className = 'coeff-item';
            item.innerHTML = `
                <span class="coeff-name">${feature} (Clase ${i})</span>
                <span class="coeff-value">${parseFloat(val).toFixed(4)}</span>
            `;
            display.appendChild(item);
        });
    });
}

// Evaluar modelo
async function evaluateModel() {
    if (!appState.modelTrained) {
        showError('Entrena un modelo primero');
        return;
    }
    
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/evaluate`);
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error);
        
        displayMetrics(data.metrics);
        displayConfusionMatrix(data.confusion_matrix, data.classes);
        showSuccess('Modelo evaluado');
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// Mostrar métricas
function displayMetrics(metrics) {
    document.getElementById('metricsDisplay').style.display = 'block';
    document.getElementById('accuracy').textContent = (metrics.accuracy * 100).toFixed(2) + '%';
    document.getElementById('precision').textContent = (metrics.precision * 100).toFixed(2) + '%';
    document.getElementById('recall').textContent = (metrics.recall * 100).toFixed(2) + '%';
    document.getElementById('f1score').textContent = (metrics.f1_score * 100).toFixed(2) + '%';
}

// Mostrar matriz de confusión
function displayConfusionMatrix(matrix, classes) {
    document.getElementById('confusionSection').style.display = 'block';

    const container = document.getElementById('confusionChart');
    container.innerHTML = '';

    const maxVal = Math.max(...matrix.flat());

    const table = document.createElement('div');
    table.className = 'cm-wrapper';

    classes.forEach((actual, i) => {
        const row = document.createElement('div');
        row.className = 'cm-row';

        const label = document.createElement('div');
        label.className = 'cm-cell cm-label';
        label.textContent = `Clase ${actual}`;
        row.appendChild(label);

        classes.forEach((predicted, j) => {
            const val = matrix[i][j];
            const intensity = maxVal > 0 ? val / maxVal : 0;

            const cell = document.createElement('div');
            cell.className = 'cm-cell cm-value';
            cell.textContent = val;

            const hue = 250;
            const light = 92 - intensity * 55;
            cell.style.background = `hsl(${hue}, 65%, ${light}%)`;
            cell.style.color = intensity > 0.5 ? '#fff' : '#2d3436';
            cell.style.fontWeight = intensity > 0.7 ? '700' : '500';

            row.appendChild(cell);
        });

        table.appendChild(row);
    });

    const header = document.createElement('div');
    header.className = 'cm-header-bar';
    const predLabel = document.createElement('span');
    predLabel.textContent = 'Predicha →';
    header.appendChild(predLabel);
    table.insertBefore(header, table.firstChild);

    container.appendChild(table);
}

// Mostrar formulario de predicción
function showPredictionForm() {
    const form = document.getElementById('predictionForm');
    form.style.display = 'block';
    
    const inputDiv = document.getElementById('featuresInput');
    inputDiv.innerHTML = '';
    
    appState.featureNames.forEach((name, idx) => {
        const group = document.createElement('div');
        group.className = 'feature-input-group';
        group.innerHTML = `
            <label>${name}</label>
            <input 
                type="number" 
                id="feature${idx}" 
                placeholder="0.0" 
                step="0.01"
                class="input-field"
            >
        `;
        inputDiv.appendChild(group);
    });
}

// Hacer predicción
async function makePrediction() {
    if (!appState.modelTrained) {
        showError('Entrena un modelo primero');
        return;
    }
    
    const features = [];
    for (let i = 0; i < appState.featureCount; i++) {
        const val = parseFloat(document.getElementById(`feature${i}`).value);
        if (isNaN(val)) {
            showError(`Valor inválido en feature ${i}`);
            return;
        }
        features.push(val);
    }
    
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features })
        });
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error);
        
        displayPrediction(data.prediction, data.probabilities, data.classes);
        showSuccess('Predicción realizada');
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// Mostrar predicción
function displayPrediction(prediction, probabilities, classes) {
    document.getElementById('predictionResult').style.display = 'block';
    document.getElementById('predictionClass').textContent = prediction;
    
    const probDisplay = document.getElementById('probabilitiesDisplay');
    probDisplay.innerHTML = '<strong>Probabilidades por Clase:</strong>';
    
    probabilities.forEach((prob, idx) => {
        const percentage = (prob * 100).toFixed(2);
        const bar = document.createElement('div');
        bar.className = 'probability-bar';
        bar.innerHTML = `
            <label>Clase ${classes[idx]}</label>
            <div class="bar-container">
                <div class="bar-fill" style="width: ${percentage}%">
                    ${percentage}%
                </div>
            </div>
        `;
        probDisplay.appendChild(bar);
    });
}

// Listeners de eventos
document.addEventListener('DOMContentLoaded', () => {
    // Dataset buttons
    document.getElementById('loadIris').addEventListener('click', () => loadDataset('iris'));
    document.getElementById('loadWine').addEventListener('click', () => loadDataset('wine'));
    document.getElementById('loadDigits').addEventListener('click', () => loadDataset('digits'));
    
    // CSV upload
    document.getElementById('uploadCSV').addEventListener('click', uploadCSV);
    
    // Train
    document.getElementById('trainBtn').addEventListener('click', trainModel);
    
    // Evaluate
    document.getElementById('evaluateBtn').addEventListener('click', evaluateModel);
    
    // Predict
    document.getElementById('predictBtn').addEventListener('click', makePrediction);
});
