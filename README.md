# Clasificador de Regresión Logística 🤖

Aplicación web para entrenar, evaluar y usar modelos de clasificación basados en regresión logística.

## Características

✅ Carga de datasets de ejemplo (Iris, Wine)  
✅ Carga de archivos CSV personalizados  
✅ Entrenamiento de modelos de regresión logística  
✅ Evaluación con métricas de clasificación (Accuracy, Precision, Recall, F1)  
✅ Matriz de confusión interactiva  
✅ Predicciones en nuevos datos  
✅ Visualización de coeficientes del modelo  

## Stack Tecnológico

**Backend:**
- Flask (Python web framework)
- scikit-learn (Machine Learning)
- pandas, numpy (Data processing)

**Frontend:**
- HTML5, CSS3, JavaScript (Vanilla)
- Chart.js (Visualizaciones)

## Estructura del Proyecto

```
proyecto-ml/
├── app.py                 # Backend Flask
├── requirements.txt       # Dependencias Python
├── Procfile              # Configuración para deploy
├── .gitignore
├── templates/
│   └── index.html        # Frontend (HTML)
└── static/
    ├── style.css         # Estilos CSS
    └── app.js            # Lógica del frontend
```

## Instalación Local

### 1. Clonar el repositorio
```bash
git clone <tu-repo>
cd proyecto-ml
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación
```bash
python app.py
```

La aplicación estará disponible en `http://localhost:5000`

## Uso

### 1. Carga de Datos
- Selecciona un dataset de ejemplo (Iris o Wine)
- O carga tu propio CSV especificando la columna target

### 2. Entrenar Modelo
- Ajusta parámetros (max_iter)
- Haz clic en "Entrenar Modelo"
- Visualiza coeficientes y métricas de entrenamiento

### 3. Evaluar
- Haz clic en "Evaluar Modelo"
- Obtén métricas de desempeño en test set
- Visualiza la matriz de confusión

### 4. Predecir
- Ingresa valores para cada feature
- Obtén predicción y probabilidades por clase

## Despliegue en la Nube

### Opción 1: Render (Recomendado - Gratis hasta cierto punto)

1. **Preparar el repositorio:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Subir a GitHub**
   ```bash
   git remote add origin https://github.com/tu-usuario/tu-repo
   git push -u origin main
   ```

3. **En Render.com:**
   - Ve a https://render.com
   - Conecta tu GitHub
   - Crea nuevo Web Service
   - Rama: main
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`

### Opción 2: Railway (También gratis con GitHub)

1. Ve a https://railway.app
2. Conecta tu repositorio GitHub
3. Railway detectará Flask automáticamente
4. Deploy automático

### Opción 3: Heroku (requiere tarjeta)

1. **Instalar Heroku CLI**
2. **Login:**
   ```bash
   heroku login
   ```

3. **Crear app:**
   ```bash
   heroku create tu-app-name
   ```

4. **Deploy:**
   ```bash
   git push heroku main
   ```

## API Endpoints

### GET `/api/datasets`
Retorna lista de datasets disponibles

### POST `/api/load-dataset/<dataset_id>`
Carga un dataset de ejemplo (iris, wine)

### POST `/api/upload-csv`
Carga datos desde archivo CSV
- Form params: `file`, `target_column`

### POST `/api/train`
Entrena el modelo
- JSON: `{"max_iter": 1000}`

### GET `/api/evaluate`
Evalúa el modelo en test set
- Retorna: métricas, matriz de confusión

### POST `/api/predict`
Realiza predicción
- JSON: `{"features": [1.5, 2.3, ...]}`

### GET `/api/model-info`
Retorna información del modelo entrenado

## Conceptos Implementados

### Regresión Logística
- Algoritmo de clasificación supervisado
- Usa función sigmoidea para probabilidades
- Optimizado con descenso de gradiente

### Separación Train/Test
- 80% entrenamiento, 20% test
- Evita overfitting

### Normalización
- StandardScaler normaliza features
- Media=0, Varianza=1

### Métricas de Clasificación
- **Accuracy:** Proporción de predicciones correctas
- **Precision:** TP/(TP+FP) - Falsos positivos
- **Recall:** TP/(TP+FN) - Falsos negativos
- **F1-Score:** Media armónica de Precision y Recall

### Matriz de Confusión
- Visualiza predicciones correctas e incorrectas por clase

## Notas Importantes

⚠️ **Video Coding Prohibido:**
Este proyecto fue desarrollado usando Claude Code (herramientas IA con agentes), NO video coding.

✅ **Arquitectura:**
- Backend desacoplado del frontend (API REST)
- Fácil de entender y modificar
- Escalable para agregar más modelos

🎓 **Aprendizaje:**
Entender cada componente es clave para la sustentación. Lee el código y experimenta.

## Estructura de Respuestas API

### Éxito
```json
{
  "success": true,
  "data": { ... }
}
```

### Error
```json
{
  "error": "Descripción del error"
}
```

## Preguntas Frecuentes

**¿Puedo cambiar el modelo a SVM o Random Forest?**
Sí, modifica `app.py` importando el modelo y cambiando el entrenamiento.

**¿Cómo agrego más datasets?**
Añade un nuevo bloque `elif` en la función `load_dataset()` en `app.py` o carga CSV personalizado.

**¿El modelo se guarda entre sesiones?**
No actualmente. Para persistencia, usa `joblib` o `pickle`.

**¿Funciona con multiclase?**
Sí, scikit-learn maneja automáticamente OvR (One vs Rest).

## Requisitos para la Sustentación

Debes entender y poder explicar:

1. ✅ Cómo se estructura el problema de clasificación
2. ✅ Separación train/test y por qué es importante
3. ✅ Cómo funciona regresión logística
4. ✅ Interpretación de coeficientes
5. ✅ Métricas de evaluación
6. ✅ Cómo usar el modelo para predicciones
7. ✅ Arquitectura de la aplicación
8. ✅ Cómo se desplegó en la nube

## Licencia

Este proyecto es para fines educativos.

---

**Autor:** Tu Nombre  
**Curso:** Inteligencia Artificial y Aprendizaje de Máquina  
**Docente:** PhD Jorge Rudas  
**Universidad:** Unicaribe
