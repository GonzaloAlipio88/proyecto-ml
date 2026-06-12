// estado de la app
let estadoApp = {
    dataLoaded: false,
    modelTrained: false,
    nombresFeatures: [],
    cantidadFeatures: 0,
    tipoModelo: 'logistic'
};

// funciones auxiliares
const mostrarLoading = () => document.getElementById('loading').style.display = 'flex';
const ocultarLoading = () => document.getElementById('loading').style.display = 'none';

function mostrarError(mensaje) {
    const errorBox = document.getElementById('errorBox');
    document.getElementById('errorMessage').textContent = mensaje;
    errorBox.style.display = 'block';
    setTimeout(() => errorBox.style.display = 'none', 5000);
}

function mostrarSuccess(mensaje) {
    console.log(mensaje);
}

const API_URL = '/api';

// cargar dataset de ejemplo
async function loadDataset(datasetId) {
    mostrarLoading();
    try {
        const response = await fetch(`${API_URL}/load-dataset/${datasetId}`, {
            method: 'POST'
        });
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error);
        
        estadoApp.dataLoaded = true;
        estadoApp.nombresFeatures = data.data_info.feature_names;
        estadoApp.cantidadFeatures = data.data_info.features;
        
        actualizarInfo(data.data_info);
        mostrarSuccess(`Dataset ${datasetId} cargado`);
    } catch (error) {
        mostrarError(error.message);
    }
    
    ocultarLoading();
}

// subir csv
async function uploadCSV() {
    const file = document.getElementById('csvFile').files[0];
    if (!file) {
        mostrarError('Selecciona un archivo CSV');
        return;
    }
    
    mostrarLoading();
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('target_column', document.getElementById('targetCol').value);
        
        const response = await fetch(`${API_URL}/upload-csv`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error);
        
        estadoApp.dataLoaded = true;
        estadoApp.nombresFeatures = data.data_info.feature_names;
        estadoApp.cantidadFeatures = data.data_info.features;
        
        actualizarInfo(data.data_info);
        mostrarSuccess('CSV cargado correctamente');
    } catch (error) {
        mostrarError(error.message);
    }
    
    ocultarLoading();
}

// actualizar info
function actualizarInfo(info) {
    document.getElementById('dataInfo').style.display = 'block';
    document.getElementById('samplesCount').textContent = info.samples;
    document.getElementById('featuresCount').textContent = info.features;
    document.getElementById('trainTestSplit').textContent = info.train_size + ' / ' + info.test_size;
    document.getElementById('classesCount').textContent = info.classes;
}

// entrenar modelo
async function trainModel() {
    if (!estadoApp.dataLoaded) {
        mostrarError('Carga datos primero');
        return;
    }
    
    mostrarLoading();
    try {
        const maxIter = document.getElementById('maxIter').value;
        const modelType = document.getElementById('modelType').value;
        estadoApp.tipoModelo = modelType;
        
        const response = await fetch(`${API_URL}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                max_iter: parseInt(maxIter),
                model_type: modelType
            })
        });
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error);
        
        estadoApp.modelTrained = true;
        
        document.getElementById('trainStatus').style.display = 'block';
        
        if (modelType === 'linear') {
            document.getElementById('trainMessage').textContent = 
                'Modelo lineal entrenado. MSE: ' + data.training_metrics.mse.toFixed(4) + 
                ', RMSE: ' + data.training_metrics.rmse.toFixed(4);
        } else {
            document.getElementById('trainMessage').textContent = 
                'Modelo entrenado. Accuracy en train: ' + (data.training_metrics.accuracy * 100).toFixed(2) + '%';
        }
        
        // cargar info del modelo entrenado
        await cargarInfoModelo();
        
        // mostrar formulario
        mostrarFormularioPrediccion();
        
        mostrarSuccess('Modelo entrenado exitosamente');
    } catch (error) {
        mostrarError(error.message);
    }
    
    ocultarLoading();
}

// cargar info del modelo
async function cargarInfoModelo() {
    try {
        const response = await fetch(`${API_URL}/model-info`);
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error);
        
        const info = data.model_info;
        document.getElementById('modelInfo').style.display = 'block';
        document.getElementById('noModel').style.display = 'none';
        document.getElementById('modelClasses').textContent = info.classes ? info.classes.join(', ') : 'N/A (regresion)';
        document.getElementById('modelFeatures').textContent = info.n_features;
        
        // mostrar coeficientes
        displayCoefficients(info.feature_names, info.coefficients);
    } catch (error) {
        console.error('Error al cargar info del modelo:', error);
    }
}

// mostrar coeficientes
function displayCoefficients(nombres, coeficientes) {
    const display = document.getElementById('coefficientsDisplay');
    display.innerHTML = '<strong>Coeficientes del Modelo:</strong>';
    
    for (let idx = 0; idx < coeficientes.length; idx++) {
        const coef = coeficientes[idx];
        const feature = nombres[idx] || 'Caracteristica ' + idx;
        
        if (Array.isArray(coef)) {
            for (let i = 0; i < coef.length; i++) {
                const item = document.createElement('div');
                item.className = 'coeff-item';
                item.innerHTML = 
                    '<span class="coeff-name">' + feature + ' (Clase ' + i + ')</span>' +
                    '<span class="coeff-value">' + parseFloat(coef[i]).toFixed(4) + '</span>';
                display.appendChild(item);
            }
        } else {
            const item = document.createElement('div');
            item.className = 'coeff-item';
            item.innerHTML = 
                '<span class="coeff-name">' + feature + '</span>' +
                '<span class="coeff-value">' + parseFloat(coef).toFixed(4) + '</span>';
            display.appendChild(item);
        }
    }
}

// evaluar modelo
async function evaluateModel() {
    if (!estadoApp.modelTrained) {
        mostrarError('Entrena un modelo primero');
        return;
    }
    
    mostrarLoading();
    try {
        const response = await fetch(`${API_URL}/evaluate`);
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error);
        
        if (data.model_type === 'linear') {
            mostrarMetricasRegresion(data.metrics);
            document.getElementById('confusionSection').style.display = 'none';
        } else {
            displayMetrics(data.metrics);
            mostrarMatrizConfusion(data.confusion_matrix, data.classes);
        }
        
        mostrarSuccess('Modelo evaluado');
    } catch (error) {
        mostrarError(error.message);
    }
    
    ocultarLoading();
}

// mostrar metricas de clasificacion
function displayMetrics(metrics) {
    document.getElementById('metricsDisplay').style.display = 'block';
    document.getElementById('classMetrics').style.display = 'block';
    document.getElementById('regMetrics').style.display = 'none';
    document.getElementById('accuracy').textContent = (metrics.accuracy * 100).toFixed(2) + '%';
    document.getElementById('precision').textContent = (metrics.precision * 100).toFixed(2) + '%';
    document.getElementById('recall').textContent = (metrics.recall * 100).toFixed(2) + '%';
    document.getElementById('f1score').textContent = (metrics.f1_score * 100).toFixed(2) + '%';
}

// mostrar metricas de regresion
function mostrarMetricasRegresion(metrics) {
    document.getElementById('metricsDisplay').style.display = 'block';
    document.getElementById('classMetrics').style.display = 'none';
    document.getElementById('regMetrics').style.display = 'block';
    document.getElementById('mse').textContent = metrics.mse.toFixed(4);
    document.getElementById('rmse').textContent = metrics.rmse.toFixed(4);
    document.getElementById('mae').textContent = metrics.mae.toFixed(4);
    document.getElementById('r2').textContent = metrics.r2.toFixed(4);
}

// mostrar matriz de confusion como heatmap
function mostrarMatrizConfusion(matrix, classes) {
    document.getElementById('confusionSection').style.display = 'block';

    const container = document.getElementById('confusionChart');
    container.innerHTML = '';

    var maxVal = 0;
    for (var i = 0; i < matrix.length; i++) {
        for (var j = 0; j < matrix[i].length; j++) {
            if (matrix[i][j] > maxVal) maxVal = matrix[i][j];
        }
    }

    const tabla = document.createElement('div');
    tabla.className = 'cm-wrapper';

    for (var i = 0; i < classes.length; i++) {
        const fila = document.createElement('div');
        fila.className = 'cm-row';

        const label = document.createElement('div');
        label.className = 'cm-cell cm-label';
        label.textContent = 'Clase ' + classes[i];
        fila.appendChild(label);

        for (var j = 0; j < classes.length; j++) {
            const val = matrix[i][j];
            const intensidad = maxVal > 0 ? val / maxVal : 0;

            const celda = document.createElement('div');
            celda.className = 'cm-cell cm-value';
            celda.textContent = val;

            const luz = 92 - intensidad * 55;
            celda.style.background = 'hsl(250, 65%, ' + luz + '%)';
            celda.style.color = intensidad > 0.5 ? '#fff' : '#2d3436';
            celda.style.fontWeight = intensidad > 0.7 ? '700' : '500';

            fila.appendChild(celda);
        }

        tabla.appendChild(fila);
    }

    const header = document.createElement('div');
    header.className = 'cm-header-bar';
    header.textContent = 'Predicha →';
    tabla.insertBefore(header, tabla.firstChild);

    container.appendChild(tabla);
}

// mostrar formulario de prediccion
function mostrarFormularioPrediccion() {
    document.getElementById('predictionForm').style.display = 'block';
    
    const inputDiv = document.getElementById('featuresInput');
    inputDiv.innerHTML = '';
    
    for (var i = 0; i < estadoApp.nombresFeatures.length; i++) {
        const group = document.createElement('div');
        group.className = 'feature-input-group';
        group.innerHTML = 
            '<label>' + estadoApp.nombresFeatures[i] + '</label>' +
            '<input type="number" id="feature' + i + '" placeholder="0.0" step="0.01" class="input-field">';
        inputDiv.appendChild(group);
    }
}

// hacer prediccion
async function makePrediction() {
    if (!estadoApp.modelTrained) {
        mostrarError('Entrena un modelo primero');
        return;
    }
    
    const features = [];
    for (var i = 0; i < estadoApp.cantidadFeatures; i++) {
        const val = parseFloat(document.getElementById('feature' + i).value);
        if (isNaN(val)) {
            mostrarError('Valor invalido en feature ' + i);
            return;
        }
        features.push(val);
    }
    
    mostrarLoading();
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: features })
        });
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error);
        
        mostrarPrediccion(data.prediction, data.probabilities, data.classes);
        mostrarSuccess('Prediccion realizada');
    } catch (error) {
        mostrarError(error.message);
    }
    
    ocultarLoading();
}

// mostrar prediccion
function mostrarPrediccion(prediction, probabilities, classes) {
    document.getElementById('predictionResult').style.display = 'block';
    document.getElementById('predictionClass').textContent = prediction;
    
    const probDisplay = document.getElementById('probabilitiesDisplay');
    
    if (probabilities && classes) {
        probDisplay.innerHTML = '<strong>Probabilidades por Clase:</strong>';
        for (var i = 0; i < probabilities.length; i++) {
            const prob = probabilities[i];
            const percentage = (prob * 100).toFixed(2);
            const bar = document.createElement('div');
            bar.className = 'probability-bar';
            bar.innerHTML = 
                '<label>Clase ' + classes[i] + '</label>' +
                '<div class="bar-container">' +
                    '<div class="bar-fill" style="width: ' + percentage + '%">' +
                        percentage + '%' +
                    '</div>' +
                '</div>';
            probDisplay.appendChild(bar);
        }
    } else {
        probDisplay.innerHTML = '';
    }
}

// listeners de eventos
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('loadIris').addEventListener('click', function() { loadDataset('iris'); });
    document.getElementById('loadWine').addEventListener('click', function() { loadDataset('wine'); });
    document.getElementById('loadDigits').addEventListener('click', function() { loadDataset('digits'); });
    
    document.getElementById('uploadCSV').addEventListener('click', uploadCSV);
    
    document.getElementById('trainBtn').addEventListener('click', trainModel);
    
    document.getElementById('evaluateBtn').addEventListener('click', evaluateModel);
    
    document.getElementById('predictBtn').addEventListener('click', makePrediction);
});
