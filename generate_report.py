# Script para generar el reporte HTML completo por Gonzalo Muñoz Olate @gmolate 2025
import os
import re
import datetime

# --- Configuración Global ---
OUTPUT_DIR = 'analisis_resultados'
METRICS_FILE = os.path.join(OUTPUT_DIR, 'resumen_metricas.txt')
REPORT_HTML_FILE = os.path.join(OUTPUT_DIR, 'reporte_final.html')

# --- Abstract del Paper (Draft) ---
PAPER_ABSTRACT = """
<div class="paper-abstract">
    <h2>Resumen</h2>
    <p><strong>Introducción:</strong> La hipofunción oral en adultos mayores chilenos es una condición prevalente y subdiagnosticada que impacta directamente la calidad de vida, estado nutricional y comorbilidades sistémicas. El EDePAM, aunque exhaustivo, presenta desafíos logísticos en atención primaria.</p>
    
    <p><strong>Objectivo:</strong> Desarrollar y validar modelos de Machine Learning para predecir hipofunción oral usando datos del EDePAM en población rural chilena, optimizando el screening clínico.</p>
    
    <p><strong>Metodo:</strong> Análisis retrospectivo de {total_pacientes} pacientes con datos basales y seguimiento. Implementamos cuatro algoritmos: Random Forest, XGBoost, SVM y Regresión Logística con validación cruzada y manejo de desbalance (SMOTE).</p>
    
    <p><strong>Resultados:</strong> Los modelos <strong>Random Forest y XGBoost mostraron el rendimiento más alto</strong>, con un Recall de Hipofunción idéntico de <strong>{rf_recall}</strong>. El análisis de explicabilidad (XAI) identificó a las variables del constructo periodontal, específicamente la <strong>extensión y el estado</strong>, como los predictores dominantes. El análisis post-intervención reveló mejoras significativas en calidad de vida (p={wilcoxon_p}).</p>
    
    <p><strong>Conclusión:</strong> Los modelos ML pueden predecir efectivamente la hipofunción oral. Proponemos el "EDePA-r" (Exámen dental preventivo del adulto resumido) como herramienta de screening optimizada para APS.</p>
</div>
"""

# --- Resumen del Estudio (README Integration) ---
STUDY_SUMMARY = """
<div class="study-summary">
    <h3>Resumen del Estudio EDePAM-ML</h3>
    <ul>
        <li><strong>Población:</strong> Adultos atendidos en hospital de Los Vilos, Chile</li>
        <li><strong>Innovación:</strong> Primer modelo ML para hipofunción oral en población chilena</li>
        <li><strong>EDePA-r Propuesto:</strong> Versión abreviada con top 5 predictores (periodontal_grado, fuerza_oclusal, diadococinesia, función_masticatoria, edad) para uso rápido en APS</li>
        <li><strong>Validación:</strong> Datos longitudinales pre/post tratamiento confirman efectividad clínica</li>
    </ul>
</div>
"""
# --- Glosario de terminos (glossary) ---
GLOSSARY_HTML = """
    <section class="glossary">
        <h2>Glosario de Términos Técnicos</h2>
        <dl>
            <dt><strong>XAI (Explainable Artificial Intelligence)</strong></dt>
            <dd>Acrónimo de <em>Explainable Artificial Intelligence</em> (Inteligencia Artificial Explicable). Refiere al conjunto de métodos y técnicas cuyo objetivo es dotar de transparencia e interpretabilidad a los modelos de machine learning, permitiendo comprender el razonamiento subyacente a sus predicciones.</dd>

            <dt><strong>SHAP (SHapley Additive exPlanations)</strong></dt>
            <dd>Técnica de XAI que explica predicciones individuales asignando a cada variable un valor de importancia (valor SHAP) que representa su contribución específica a dicha predicción. Permite un análisis de atribución a nivel local (por paciente).</dd>

            <dt><strong>Recall (Sensibilidad)</strong></dt>
            <dd>Métrica de evaluación crítica en contextos clínicos. Mide la proporción de casos positivos reales que son correctamente identificados por el modelo. Un <strong>Recall</strong> elevado es fundamental para minimizar la tasa de falsos negativos.</dd>

            <dt><strong>Accuracy (Precisión Global)</strong></dt>
            <dd>Mide el porcentaje total de predicciones correctas sobre el conjunto de datos. Puede ser un indicador poco fiable en problemas con clases desbalanceadas.</dd>

            <dt><strong>AUC (Area Under the Curve)</strong></dt>
            <dd>Acrónimo de <em>Area Under the Curve</em> (Área Bajo la Curva ROC). Representa la capacidad de un modelo para discriminar entre las clases positiva y negativa. Un valor de 1.0 indica un clasificador perfecto.</dd>

            <dt><strong>F1-Score</strong></dt>
            <dd>Media armónica entre Precisión y Recall. Proporciona una medida única que equilibra la tasa de falsos positivos y falsos negativos.</dd>

            <dt><strong>CV (Cross-Validation)</strong></dt>
            <dd>Técnica de Validación Cruzada que evalúa la capacidad de generalización de un modelo para obtener una estimación más robusta de su rendimiento.</dd>

            <dt><strong>N/A (Not Available)</strong></dt>
            <dd>Indica que no hay datos disponibles para la métrica en cuestión.</dd>
        </dl>
    </section>
    """

# --- Explicaciones de Algoritmos ---

ALGORITHM_EXPLANATIONS = {
    'RF': """
    <div class="explanation">
        <h4>Random Forest: El Guardián del conjunto</h4>
        <p>Combina múltiples árboles de decisión, generando un consenso final para crear predicciones robustas. Ideal para datos médicos por su capacidad de manejar interacciones complejas entre variables clínicas como periodontal-oclusal-masticatorio sin sobreajuste.</p>
    </div>
    """,
    'SVM': """
    <div class="explanation">
        <h4>SVM: El Guardian de Fronteras</h4>
        <p>Support Vector Machine destaca funciona creando una "frontera" o línea de separación lo más clara posible entre dos grupos de datos permite capturar relaciones no-lineales como la curva caries-oclusal, crucial para casos borderline periodontales.</p>
    </div>
    """,
    'XGB': """
    <div class="explanation">
        <h4>XGBoost: Potenciación del Gradiente o Reforzamiento por Gradiente</h4>
        <p>"Algoritmo de boosting que aprende secuencialmente de sus errores para modelar patrones complejos (no-lineales) en la funcionalidad oral, usando variables como el recuento de caries."</p>
    </div>
    """,
    'LR': """
    <div class="explanation">
        <h4>Regresión Logística: La Base Interpretable</h4>
        <p>Modelo baseline que proporciona odds ratios interpretables para clínicos. Fundamental para entender la contribución individual de cada factor (edad, periodontal, protésico) en la probabilidad de hipofunción.</p>
    </div>
    """
}

def parse_comprehensive_metrics(content):
    """
    Parsea el archivo resumen_metricas.txt, que tiene un formato limpio.
    """
    results = {'models': {}, 'stats': {}, 'images': {}}
    
    # --- Parsear Métricas de Modelos V4 ---
    model_acronyms = {'rf': 'Random Forest', 'svm': 'Support Vector Machine', 'lr': 'Logistic Regression', 'xgb': 'XGBoost'}
    
    for key, name in model_acronyms.items():
        # Crear un patrón de regex que capture todo el bloque de un modelo
        pattern = rf"{name} \({key.upper()}\):.*?- Accuracy: ([\d\.]+).*?- F1-Score \(Weighted\): ([\d\.]+).*?- AUC: ([\d\.]+).*?- Recall Hipofuncion: ([\d\.]+)"
        
        match = re.search(pattern, content, re.DOTALL)
        if match:
            results['models'][f'v4_{key}'] = {
                'accuracy': match.group(1),
                'f1': match.group(2),
                'auc': match.group(3),
                'recall_hipo': match.group(4)
            }

    # --- Parsear Tests Estadísticos ---
    wilcoxon_match = re.search(r"Calidad de Vida \(EQ-5D\) - Wilcoxon p-value: ([\d\.]+)", content)
    ttest_match = re.search(r"Grado Periodontal - Paired t-test p-value: ([\d\.]+|nan)", content)
    
    results['stats'] = {
        'wilcoxon_p': wilcoxon_match.group(1) if wilcoxon_match else 'N/A',
        'ttest_p': ttest_match.group(1) if ttest_match else 'N/A'
    }
    
    # --- Verificar Existencia de Imágenes ---
    image_files = [
        # Gráficos de Random Forest
        'cm_v4_rf_RF.png', 'shap_v4_rf_RF.png', 'imp_v4_rf_RF.png',
        
        # Gráficos de SVM
        'cm_v4_svm_SVM.png', 'imp_v4_svm_SVM.png',
        
        # Gráficos de XGBoost (NUEVOS)
        'cm_v4_xgb_XGB.png', 'shap_v4_xgb_XGB.png', 'imp_v4_xgb_XGB.png',
        
        # Gráfico Pre/Post
        'cambios_func.png',
        
        # Gráfico de Correlación (NUEVO)
        'correlacion_heatmap.png'
    ]
    
    results['images'] = {}
    for img in image_files:
        # Crea una clave más simple para buscar, ej: 'cmv4rfRF'
        img_key = img.replace('.png', '').replace('_', '')
        if os.path.exists(os.path.join(OUTPUT_DIR, img)):
            results['images'][img_key] = img
        else:
            results['images'][img_key] = None
            
    return results

def generate_model_section(model_data, model_name, title):
    """Generate HTML section for a specific model if data exists"""
    if not model_data or model_data.get('accuracy') == 'N/A':
        return ""
    
    return f"""
    <section class="model-section">
        <h3>{title}</h3>
        <div class="metrics-grid">
            <div class="metric">
                <span class="metric-label">Accuracy:</span>
                <span class="metric-value">{model_data.get('accuracy', 'N/A')}</span>
            </div>
            <div class="metric">
                <span class="metric-label">F1-Score:</span>
                <span class="metric-value">{model_data.get('f1', 'N/A')}</span>
            </div>
            <div class="metric">
                <span class="metric-label">AUC:</span>
                <span class="metric-value">{model_data.get('auc', 'N/A')}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Recall Hipofunción:</span>
                <span class="metric-value">{model_data.get('recall_hipo', 'N/A')}</span>
            </div>
        </div>
        {ALGORITHM_EXPLANATIONS.get(model_name.upper(), '')}
    </section>
    """

def generate_v4_comparison_table(models_data):
    """Generate V4 models comparison table with insights"""
    v4_models = ['v4_rf', 'v4_svm', 'v4_xgb', 'v4_lr']
    model_names = {'v4_rf': 'Random Forest', 'v4_svm': 'SVM', 'v4_xgb': 'XGBoost', 'v4_lr': 'Logistic Regression'}
    
    rows = []
    for model in v4_models:
        if model in models_data:
            data = models_data[model]
            rows.append(f"""
            <tr>
                <td><strong>{model_names[model]}</strong></td>
                <td>{data.get('accuracy', 'N/A')}</td>
                <td>{data.get('f1', 'N/A')}</td>
                <td>{data.get('auc', 'N/A')}</td>
                <td>{data.get('recall_hipo', 'N/A')}</td>
            </tr>
            """)
    
    insight_row = """
    <tr class="insight-row">
        <td colspan="5">
            <strong>Insight Clínico:</strong> SVM destaca en recall para casos raros de hipofunción debido a su robusta separación de fronteras. RF ofrece el mejor balance general. XGBoost captura patrones no-lineales complejos.
        </td>
    </tr>
    """
    
    return f"""
    <table class="comparison-table">
        <thead>
            <tr>
                <th>Modelo V4</th>
                <th>Accuracy</th>
                <th>F1-Score</th>
                <th>AUC</th>
                <th>Recall Hipofunción</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
            {insight_row}
        </tbody>
    </table>
    """

def generate_image_section(images, model_type):
    """Generate image sections with detailed captions for a specific model or type"""
    sections = []
    
    # --- HEATMAP ---
    if model_type == 'heatmap' and images.get('correlacionheatmap'):
        sections.append(f"""
        <div class="figure">
            <h4>Mapa de Calor de Correlaciones</h4>
            <img src="{images['correlacionheatmap']}" alt="Correlation Heatmap">
            <p class="caption">Este mapa valida la consistencia interna del instrumento, mostrando fuertes correlaciones (rojo) entre las variables funcionales clave.</p>
        </div>
        """)

    # --- RANDOM FOREST ---
    if model_type == 'rf':
        if images.get('cmv4rfRF'):
            sections.append(f"""
            <div class="figure">
                <h4>Matriz de Confusión - Random Forest</h4>
                <img src="{images['cmv4rfRF']}" alt="Confusion Matrix RF">
                <p class="caption">Esta matriz muestra la excelente capacidad de RF para minimizar falsos negativos en hipofunción, ideal para screening clínico preventivo.</p>
            </div>
            """)
        if images.get('impv4rfRF'):
            sections.append(f"""
            <div class="figure">
                <h4>Importancia de Features - Random Forest</h4>
                <img src="{images['impv4rfRF']}" alt="Feature Importance RF">
                <p class="caption">El gráfico confirma que las variables periodontales y de función oral dominan las predicciones del modelo RF.</p>
            </div>
            """)
        if images.get('shapv4rfRF'):
            sections.append(f"""
            <div class="figure">
                <h4>Análisis SHAP - Random Forest</h4>
                <img src="{images['shapv4rfRF']}" alt="SHAP Plot RF">
                <p class="caption">El análisis SHAP revela que periodontal_grado domina las predicciones, seguido por fuerza_oclusal. Los valores rojos (altos) impulsan hacia hipofunción.</p>
            </div>
            """)

    # --- XGBOOST ---
    if model_type == 'xgb':
        if images.get('cmv4xgbXGB'):
            sections.append(f"""
            <div class="figure">
                <h4>Matriz de Confusión - XGBoost</h4>
                <img src="{images['cmv4xgbXGB']}" alt="Confusion Matrix XGBoost">
                <p class="caption">XGBoost muestra un balance sólido entre precisión y recall, capturando patrones no-lineales complejos en los datos.</p>
            </div>
            """)
        if images.get('impv4xgbXGB'):
            sections.append(f"""
            <div class="figure">
                <h4>Importancia de Features - XGBoost</h4>
                <img src="{images['impv4xgbXGB']}" alt="Feature Importance XGBoost">
                <p class="caption">El gradient boosting de XGBoost identifica sutiles interacciones entre variables que otros modelos podrían pasar por alto.</p>
            </div>
            """)
        if images.get('shapv4xgbXGB'):
            sections.append(f"""
            <div class="figure">
                <h4>Análisis SHAP - XGBoost</h4>
                <img src="{images['shapv4xgbXGB']}" alt="SHAP Plot XGBoost">
                <p class="caption">SHAP para XGBoost revela cómo el modelo pondera cada característica, crucial para la interpretabilidad clínica.</p>
            </div>
            """)

    # --- SVM ---
    if model_type == 'svm':
        if images.get('cmv4svmSVM'):
            sections.append(f"""
            <div class="figure">
                <h4>Matriz de Confusión - SVM</h4>
                <img src="{images['cmv4svmSVM']}" alt="Confusion Matrix SVM">
                <p class="caption">Esta matriz demuestra la capacidad de SVM como "guardian de fronteras" para casos periodontales borderline, minimizando falsos negativos.</p>
            </div>
            """)
        if images.get('impv4svmSVM'):
            sections.append(f"""
            <div class="figure">
                <h4>Importancia de Features - SVM</h4>
                <img src="{images['impv4svmSVM']}" alt="Feature Importance SVM">
                <p class="caption">La importancia por permutación en SVM destaca las variables más críticas para la separación entre clases funcionales.</p>
            </div>
            """)
            
    return ''.join(sections)

def generate_complete_html_report(data):
    """Generate the complete HTML report"""
    
    # Extract key metrics for abstract
    rf_recall = data['models'].get('v4_rf', {}).get('recall_hipo', 'N/A')
    svm_recall = data['models'].get('v4_svm', {}).get('recall_hipo', 'N/A')
    wilcoxon_p = data['stats'].get('wilcoxon_p', 'N/A')
    ttest_p = data['stats'].get('ttest_p', 'N/A')
    
    # Format abstract with actual values
    formatted_abstract = PAPER_ABSTRACT.format(
        total_pacientes="40",  # Adjust based on actual data
        rf_recall=rf_recall,
        svm_recall=svm_recall,
        wilcoxon_p=wilcoxon_p,
        ttest_p=ttest_p
    )
    
    html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Completo EDePAM-ML</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }}
        .study-summary {{
            background: #e3f2fd;
            border-left: 5px solid #2196f3;
            padding: 1.5rem;
            margin: 2rem 0;
            border-radius: 5px;
        }}
        .paper-abstract {{
            background: #f3e5f5;
            border-left: 5px solid #9c27b0;
            padding: 1.5rem;
            margin: 2rem 0;
            border-radius: 5px;
        }}
        .glossary {{
            background: #fff3e0;
            border-left: 5px solid #ff9800;
            padding: 1.5rem;
            margin: 2rem 0;
            border-radius: 5px;
        }}
        .glossary dl {{
            margin: 0;
        }}
        .glossary dt {{
            font-weight: bold;
            margin-top: 1rem;
            color: #e65100;
        }}
        .glossary dd {{
            margin-left: 1rem;
            margin-bottom: 1rem;
            line-height: 1.5;
        }}
        .model-section {{
            background: white;
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        .metric {{
            background: #f5f5f5;
            padding: 1rem;
            border-radius: 5px;
            text-align: center;
        }}
        .metric-label {{
            display: block;
            font-weight: bold;
            color: #666;
            margin-bottom: 0.5rem;
        }}
        .metric-value {{
            display: block;
            font-size: 1.5rem;
            color: #2196f3;
            font-weight: bold;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 2rem 0;
            background: white;
        }}
        .comparison-table th, .comparison-table td {{
            padding: 12px;
            border: 1px solid #ddd;
            text-align: center;
        }}
        .comparison-table th {{
            background: #3498db;
            color: white;
        }}
        .insight-row {{
            background: #fffbf0;
            font-style: italic;
        }}
        .explanation {{
            background: #f0f8ff;
            border-left: 3px solid #4CAF50;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
        }}
        .figure {{
            background: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .caption {{
            font-style: italic;
            color: #666;
            margin-top: 1rem;
            text-align: left;
        }}
        .footer {{
            text-align: center;
            margin-top: 3rem;
            padding: 2rem;
            background: #263238;
            color: white;
            border-radius: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Análisis Predictivo de Hipofunción Oral</h1>
        <h2>EDePAM-ML: Machine Learning</h2>
        <p><strong>Autor:</strong> Gonzalo Muñoz Olate | <strong>Hospital de Los Vilos</strong></p>
    </div>

    {formatted_abstract}
    {STUDY_SUMMARY}
    {GLOSSARY_HTML}

    <section>
        <h2>Análisis Exploratorio de Datos</h2>
        {generate_image_section(data['images'], 'heatmap')}
    </section>

    <section>
        <h2>Comparación de Modelos V4 (Datos Completos)</h2>
        {generate_v4_comparison_table(data['models'])}
    </section>

    <section>
        <h2>Análisis Detallado por Modelo</h2>
        {generate_model_section(data['models'].get('v4_rf'), 'RF', 'Random Forest - Modelo Recomendado')}
        {generate_model_section(data['models'].get('v4_svm'), 'SVM', 'Support Vector Machine - Guardian de Fronteras')}
        {generate_model_section(data['models'].get('v4_xgb'), 'XGB', 'XGBoost - Potencia del Gradient Boosting')}
        {generate_model_section(data['models'].get('v4_lr'), 'LR', 'Regresión Logística - Base Interpretable')}
    </section>

    <section>
        <h2>Visualizaciones y Análisis por Modelo</h2>
        <h3>Random Forest</h3>
        {generate_image_section(data['images'], 'rf')}
        
        <h3>XGBoost</h3>
        {generate_image_section(data['images'], 'xgb')}
        
        <h3>Support Vector Machine</h3>
        {generate_image_section(data['images'], 'svm')}
    </section>

    <section class="model-section">
        <h2>Resultados Pre/Post Tratamiento</h2>
        <div class="metrics-grid">
            <div class="metric">
                <span class="metric-label">Mejora Calidad de Vida:</span>
                <span class="metric-value">p = {wilcoxon_p}</span>
                <p>La mejora observada en la calidad de vida de los pacientes después del tratamiento es real y muy significativa. No fue una coincidencia; el tratamiento funcionó y tuvo un impacto positivo medible.</p>
            </div>
            <div class="metric">
                <span class="metric-label">Reducción Periodontal:</span>
                <span class="metric-value">p = {ttest_p}</span>
                <p>Debido a datos insuficientes en el seguimiento, no se pudo calcular si la mejora en la salud periodontal fue estadísticamente significativa. El reporte no dice que no hubo mejora, solo que no se pudo probar matemáticamente con la data disponible.</p>
            </div>
        </div>
    </section>

    <div class="footer">
        <p>Este reporte HTML es generado al ejecutar generate_report.py el {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>EDePAM-ML Project © 2025 | Codificado en python @gmolate</p>
    </div>
</body>
</html>
    """
    
    return html_content

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Iniciando generacion de reporte completo ---")
    
    if not os.path.exists(METRICS_FILE):
        print(f"ERROR: No se encontro el archivo '{METRICS_FILE}'")
        print("Ejecuta 'run_full_analysis.py' primero.")
        exit(1)
    
    # Read with robust encoding handling
    log_content = None
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            with open(METRICS_FILE, 'r', encoding=encoding) as f:
                log_content = f.read()
            print(f"Archivo leido con codificacion: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if not log_content:
        print("ERROR: No se pudo leer el archivo con ninguna codificacion")
        exit(1)
    
    # Parse comprehensive data
    parsed_data = parse_comprehensive_metrics(log_content)
    print("Datos parseados exitosamente.")
    
    # Generate complete HTML report
    html_content = generate_complete_html_report(parsed_data)
    print("Contenido HTML generado.")
    
    # Write final report
    with open(REPORT_HTML_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nReporte completo finalizado: {REPORT_HTML_FILE}")
    print("Incluye: resumen, comparacion V4, explicaciones algoritmicas, imagenes con captions")
