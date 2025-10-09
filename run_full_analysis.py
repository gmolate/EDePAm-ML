#Código creado por Gonzalo Muñoz Olate @gmolate 2025 /// gonzalo.munoz@uchile.cl

# Check for required dependencies first/revisa los requisitos primero.
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'seaborn', 'matplotlib', 'sklearn', 'xgboost', 
        'imblearn', 'shap', 'numpy', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'imblearn':
                import imblearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("ERROR: Faltan los siguientes paquetes de Python:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPara instalar todos los paquetes necesarios, ejecuta:")
        print("pip install pandas seaborn matplotlib scikit-learn xgboost imbalanced-learn shap numpy scipy statsmodels")
        print("\nO si usas conda:")
        print("conda install pandas seaborn matplotlib scikit-learn xgboost imbalanced-learn shap numpy scipy statsmodels")
        sys.exit(1)
    
    print("✓ Todas las dependencias están instaladas correctamente.")

# Check dependencies before importing anything else
check_dependencies()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import shap
import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon, ttest_rel
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Make mcnemar optional (it's in statsmodels, not scipy)
try:
    from statsmodels.stats.contingency_tables import mcnemar
    MCNEMAR_AVAILABLE = True
except ImportError:
    MCNEMAR_AVAILABLE = False
    # Simple McNemar implementation
    def mcnemar_simple(table):
        """Simple McNemar test implementation"""
        if table.shape != (2, 2):
            return None, None
        b = table.iloc[0, 1]  # off-diagonal elements
        c = table.iloc[1, 0]
        if b + c == 0:
            return None, 1.0
        # Chi-square with continuity correction
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=1)
        return chi2, p_value
    
    print("Warning: statsmodels not available. Using simple McNemar implementation.")

import warnings
warnings.filterwarnings('ignore')

# --- Configuración Global ---
EXCEL_FILE_INICIAL = 'Base datos inicial.xlsx'
EXCEL_FILE_SEGUIMIENTO = 'Base datos seguimiento.xlsx'
CSV_DIR = 'datos_csv'  # Directorio de CSVs limpios
OUTPUT_DIR = 'analisis_resultados'  # Outputs (PNGs, logs)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Definición de Nombres de Columnas Limpios para cada Pestaña (INICIAL) ---
# DIAGNOSTICO (7 columnas) - AHORA SIN 'funcionalidad_oral_final'
clean_cols_diagnostico = [
    'correlativo',
    'periodontal_caso',
    'periodontal_estado',
    'periodontal_extension',
    'periodontal_grado',
    'implantes_presencia',
    'implantes_estado'
]

# TRASPASO CUESTIONARIO (66 columnas)
clean_cols_traspaso = [
    'correlativo', 'edad', 'sexo', 'uso_ges_60', 'nivel_educacional', 'alcanza_dinero_fin_mes',
    'resultado_empam', 'diabetes', 'hba1c', 'dislipidemia', 'hipertension', 'otra_enfermedad',
    'otra_enfermedad_cual', 'tto_cancer_cyc', 'usa_farmacos', 'farmacos_cuales', 'alergias',
    'es_fumador', 'cigarros_por_dia', 'consume_alcohol', 'frecuencia_alcohol', 'consume_drogas',
    'frecuencia_drogas', 'peso_kg', 'altura_cm', 'contenido_dieta_cariogenica', 'frecuencia_dieta',
    'consumo_citricos', 'consumo_bebidas_carbonatadas', 'consumo_vino', 'consumo_sidra',
    'boca_seca_al_comer', 'necesita_liquidos_tragar', 'dificultad_tragar_general', 'cantidad_saliva',
    'protesis_maxilar', 'protesis_maxilar_extension', 'protesis_mandibular', 'protesis_mandibular_extension',
    'protesis_mandibular_material', 'satisfecho_protesis_sup', 'motivo_insatisfaccion_sup',
    'satisfecho_protesis_inf', 'motivo_insatisfaccion_inf', 'n_protesis_sup_hechas', 'n_protesis_inf_hechas',
    'usa_pasta_5000ppm', 'usa_colutorios', 'usa_pasta_1000ppm',
    'ttm_dolor_mandibula_4sem', 'ttm_dolor_masticar_4sem', 'ttm_dolor_abrir_boca_4sem',
    'ttm_mandibula_trabada_4sem', 'ttm_ruidos_articulacion_4sem', 'ttm_requiere_derivacion',
    'eat10_perdida_peso', 'eat10_comer_fuera', 'eat10_tragar_liquidos', 'eat10_tragar_solidos',
    'eat10_tragar_pastillas', 'eat10_tragar_doloroso', 'eat10_placer_comer', 'eat10_comida_pegada',
    'eat10_tos_al_comer', 'eat10_tragar_estresante', 'eat10_requiere_derivacion'
]

# CALIDAD DE VIDA BASAL (33 columnas)
clean_cols_calidad_vida = [
    'correlativo',
    'gohai_q1', 'gohai_q2', 'gohai_q3', 'gohai_q4', 'gohai_q5', 'gohai_q6',
    'gohai_q7', 'gohai_q8', 'gohai_q9', 'gohai_q10', 'gohai_q11', 'gohai_q12',
    'ohip_q1', 'ohip_q2', 'ohip_q3', 'ohip_q4', 'ohip_q5', 'ohip_q6',
    'ohip_q7', 'ohip_q8', 'ohip_q9', 'ohip_q10', 'ohip_q11', 'ohip_q12',
    'ohip_q13', 'ohip_q14',
    'eq5d_movilidad', 'eq5d_cuidado_personal', 'eq5d_actividades_habituales',
    'eq5d_dolor_malestar', 'eq5d_angustia_depresion', 'eq5d_score_paciente'
]

# FUNCIONALIDAD ORAL (12 columnas)
clean_cols_funcionalidad_oral = [
    'correlativo',
    'leake_q1', 'leake_q2', 'leake_q3', 'leake_q4', 'leake_q5',
    'leake_resultado', 'eichner', 'funcion_masticatoria',
    'fuerza_oclusal', 'diadococinesia', 'funcion_deglutoria'
]

# LESIONES MUCOSA ORAL (4 columnas)
clean_cols_lesiones_mucosa_oral = [
    'correlativo',
    'lesion_ubicacion',
    'lesion_elemental',
    'lesion_nombre_texto'
]

# COPD (33 columnas)
clean_cols_copd = [
    'correlativo',
    'diente_18', 'diente_17', 'diente_16', 'diente_15', 'diente_14', 'diente_13', 'diente_12', 'diente_11',
    'diente_21', 'diente_22', 'diente_23', 'diente_24', 'diente_25', 'diente_26', 'diente_27', 'diente_28',
    'diente_48', 'diente_47', 'diente_46', 'diente_45', 'diente_44', 'diente_43', 'diente_42', 'diente_41',
    'diente_31', 'diente_32', 'diente_33', 'diente_34', 'diente_35', 'diente_36', 'diente_37', 'diente_38'
]

# --- Definición de Nombres de Columnas Limpios para Traspaso Cuestionario (65 columnas de seguimiento) ---
clean_cols_traspaso_seguimiento = [
    'correlativo', 'edad', 'sexo', 'tiempo_seguimiento_alta', 'motivo_no_termino_tto',
    'nivel_funcionalidad_empam', 'diabetes', 'hba1c', 'dislipidemia', 'hipertension', 'otra_enfermedad',
    'otra_enfermedad_cual', 'tto_cancer_cyc', 'usa_farmacos', 'farmacos_cuales', 'alergias',
    'es_fumador', 'cigarros_por_dia', 'consume_alcohol', 'frecuencia_alcohol', 'consume_drogas',
    'frecuencia_drogas', 'peso_kg', 'altura_cm', 'contenido_dieta_cariogenica', 'frecuencia_dieta',
    'consumo_citricos', 'consumo_bebidas_carbonatadas', 'consumo_vino', 'consumo_sidra',
    'boca_seca_al_comer', 'necesita_liquidos_tragar', 'dificultad_tragar_general', 'cantidad_saliva',
    'protesis_maxilar', 'protesis_maxilar_extension', 'protesis_mandibular', 'protesis_mandibular_extension',
    'protesis_mandibular_material', 'satisfecho_protesis_sup', 'motivo_insatisfaccion_sup',
    'satisfecho_protesis_inf', 'motivo_insatisfaccion_inf', 'n_protesis_sup_hechas', 'n_protesis_inf_hechas',
    'usa_pasta_5000ppm', 'usa_colutorios', 'usa_pasta_1000ppm',
    'ttm_dolor_mandibula_4sem', 'ttm_dolor_masticar_4sem', 'ttm_dolor_abrir_boca_4sem',
    'ttm_mandibula_trabada_4sem', 'ttm_ruidos_articulacion_4sem', 'ttm_requiere_derivacion',
    'eat10_perdida_peso', 'eat10_comer_fuera', 'eat10_tragar_liquidos', 'eat10_tragar_solidos',
    'eat10_tragar_pastillas', 'eat10_tragar_doloroso', 'eat10_placer_comer', 'eat10_comida_pegada',
    'eat10_tos_al_comer', 'eat10_tragar_estresante', 'eat10_requiere_derivacion'
]

# --- Función de Carga y Limpieza de Pestañas (Generalizada) ---
def load_and_clean_sheet(excel_path, sheet_name, clean_cols):
    print(f"  Cargando y limpiando pestaña '{sheet_name}'...")
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=1) # header=1 para fila 2
    
    # Eliminar la fila descriptiva 'NO MODIFICAR...' (ahora en la primera fila del DF)
    df = df[~df[df.columns[0]].astype(str).str.contains('NO MODIFICAR', na=False)]
    
    # Asignar los nombres de columna limpios
    if len(df.columns) != len(clean_cols):
        print(f"  ADVERTENCIA en {sheet_name}: Columnas leídas ({len(df.columns)}) no coinciden con nombres definidos ({len(clean_cols)}).")
        # Intentar renombrar solo las que coincidan por posición
        new_names_map = {df.columns[i]: clean_cols[i] for i in range(min(len(df.columns), len(clean_cols)))}
        df.rename(columns=new_names_map, inplace=True)
    else:
        df.columns = clean_cols
    
    # Filtrar filas donde 'correlativo' sea nulo y convertir a int
    df.dropna(subset=['correlativo'], inplace=True)
    df['correlativo'] = pd.to_numeric(df['correlativo'], errors='coerce').astype(int)
    
    # Convertir otras columnas a tipo numérico donde sea posible
    for col in df.columns:
        if col not in ['correlativo', 'otra_enfermedad_cual', 'farmacos_cuales', 'lesion_nombre_texto', 'motivo_no_termino_tto']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"  Pestaña '{sheet_name}' limpia. {len(df)} registros.")
    return df

# --- Función para Ejecutar y Evaluar un Modelo --- (Generalizada)
def run_model_pipeline(X_data, y_data, model_type, target_name_for_report, output_prefix, use_smote=True):
    print(f"\n--- Ejecutando Modelo {model_type} para {output_prefix} ---")
    
    if len(np.unique(y_data)) < 2:
        print(f"  Error: Solo una clase en el target para {model_type}. Skip.")
        return None, None, None, None
    
    # Chequeo desbalance
    print(f"  Target en {model_type}: {np.bincount(y_data)} (ratio hipo/normal: {sum(y_data==1)/sum(y_data==0):.2f})")
    
    # Imputa ANTES de feature selection - mejorado
    X_imputed = X_data.copy()
    
    # Replace infinite values with NaN first
    X_imputed = X_imputed.replace([np.inf, -np.inf], np.nan)
    
    # Drop columns that are entirely NaN
    X_imputed = X_imputed.dropna(axis=1, how='all')
    
    # For remaining NaNs, use mean imputation, but fill with 0 if mean is also NaN
    for col in X_imputed.columns:
        if X_imputed[col].isna().any():
            mean_val = X_imputed[col].mean()
            if pd.isna(mean_val):
                X_imputed[col] = X_imputed[col].fillna(0)
            else:
                X_imputed[col] = X_imputed[col].fillna(mean_val)
    
    # Final check - replace any remaining NaNs with 0 es decir si no hay entonces asume 0
    X_imputed = X_imputed.fillna(0)
    
    print(f"  Después de limpieza: {X_imputed.shape[1]} columnas (eliminadas {X_data.shape[1] - X_imputed.shape[1]} columnas vacías)")
    
    # Feature Selection pa' high-dim (top 20)
    selector = SelectKBest(f_classif, k=min(20, X_imputed.shape[1]))
    X_selected = selector.fit_transform(X_imputed, y_data)
    selected_features = X_imputed.columns[selector.get_support()].tolist()
    X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X_imputed.index)
    print(f"  Features reducidas a {X_selected.shape[1]} (de {X_imputed.shape[1]})")
    
    # Split
    strat = True if len(y_data.unique()) > 1 and len(y_data) > 20 else False
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_data, test_size=0.3, random_state=42, stratify=y_data if strat else None)
    
    # SMOTE pa' desbalance (solo train)
    if use_smote and sum(y_train==1) > 0 and sum(y_train==0) > 0 and sum(y_train==1) < len(y_train):
        smote = SMOTE(random_state=42, k_neighbors=min(3, sum(y_train==1)-1))
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"  SMOTE: Balanceado train de {sum(y_train==1)}/{len(y_train)} hipos a {sum(y_train_res==1)}/{len(y_train_res)}")
    else:
        X_train_res, y_train_res = X_train, y_train
    
    # Modelos (con weights)
    if model_type == 'XGB':
        pos_weight = sum(y_train_res==0) / sum(y_train_res==1) if sum(y_train_res==1) > 0 else 1
        model = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=pos_weight, eval_metric='logloss', use_label_encoder=False)
    elif model_type == 'LR':
        model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=200)
    elif model_type == 'SVM':
        model = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    elif model_type == 'RF':
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    else:
        print(f"  Error: Modelo {model_type} no soportado.")
        return None, None, None, None
    
    model.fit(X_train_res, y_train_res)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1]) if hasattr(model, 'predict_proba') and len(np.unique(y_test)) > 1 else None
    recall_hipo = f1_score(y_test, y_pred, average=None)[1] if len(np.unique(y_test)) > 1 and len(f1_score(y_test, y_pred, average=None)) > 1 else None # Recall para clase 1 (hipofunción)
    
    # CV pa' small N (LOO si <30, sino 5-fold)
    cv_folds = LeaveOneOut() if len(y_data) < 30 else 5
    cv_scores = cross_val_score(model, X_selected, y_data, cv=cv_folds, scoring='f1_weighted')
    
    # Fix AUC formatting
    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    recall_hipo_str = f"{recall_hipo:.4f}" if recall_hipo is not None else "N/A"
    print(f"  {model_type} (SMOTE={use_smote}) - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_str}, CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f}), Recall Hipo: {recall_hipo_str}")
    
    # Report - Fix encoding issue
    target_names = ['Normal (0)', 'Hipofuncion (1)'] if target_name_for_report == 'funcionalidad_oral_final' else ['Normal (0)', 'Reducida (1)']
    print(f"  Reporte de Clasificacion:\n{classification_report(y_test, y_pred, target_names=target_names)}")
    
    # CM
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Hipo'], yticklabels=['Normal', 'Hipo'])
    plt.title(f'Matriz Confusión {model_type}')
    plt.savefig(os.path.join(OUTPUT_DIR, f'cm_{output_prefix}_{model_type}.png'))
    plt.clf()
    
    # Importancias
    if hasattr(model, 'feature_importances_'):
        imp_df = pd.DataFrame({'feature': X_selected.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).head(20)
        sns.barplot(x='importance', y='feature', data=imp_df)
        plt.title(f'Top 20 Features {model_type}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'imp_{output_prefix}_{model_type}.png'))
        plt.clf()
        print("  Top 5:", imp_df.head())
    elif model_type in ['SVM', 'LR']:
        # Permutation importance para SVM/LR
        imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        imp_df = pd.DataFrame({'feature': X_selected.columns, 'importance': imp.importances_mean}).sort_values('importance', ascending=False).head(20)
        sns.barplot(x='importance', y='feature', data=imp_df)
        plt.title(f'Top 20 Features {model_type} (Permutation)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'imp_{output_prefix}_{model_type}.png'))
        plt.clf()
        print("  Top 5 (permutation):", imp_df.head())
    
    # SHAP solo para XGB y RF
    if model_type in ['XGB', 'RF']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, feature_names=X_selected.columns, show=False)
        plt.savefig(os.path.join(OUTPUT_DIR, f'shap_{output_prefix}_{model_type}.png'))
        plt.clf()
        print("  SHAP guardado: Contribs por feature.")
    
    return acc, f1, auc, recall_hipo

# --- Script Principal: Orquestador del Análisis Completo ---
if __name__ == "__main__":
    print("\n--- INICIANDO ANÁLISIS COMPLETO EDEPAM ---")
    excel_path_inicial = os.path.join(os.getcwd(), EXCEL_FILE_INICIAL)
    excel_path_seguimiento = os.path.join(os.getcwd(), EXCEL_FILE_SEGUIMIENTO)
    
    if not os.path.exists(excel_path_inicial):
        print(f"ERROR: El archivo Excel '{excel_path_inicial}' no fue encontrado. Abortando.")
        import sys
        sys.exit(1)
    if not os.path.exists(excel_path_seguimiento):
        print(f"ERROR: El archivo Excel '{excel_path_seguimiento}' no fue encontrado. Abortando.")
        import sys
        sys.exit(1)
    
    try:
        # --- PASO 1: Cargar y Limpiar Todas las Pestañas de Datos Iniciales ---
        print("\n[PASO 1/5] Cargando y Limpiando todas las pestañas de Base datos inicial.xlsx...")
        df_diag_ini = load_and_clean_sheet(excel_path_inicial, 'DIAGNOSTICO', clean_cols_diagnostico)
        df_trasp_ini = load_and_clean_sheet(excel_path_inicial, 'Traspaso Cuestionario', clean_cols_traspaso)
        df_cal_ini = load_and_clean_sheet(excel_path_inicial, 'Calidad de vida Basal', clean_cols_calidad_vida)
        df_func_ini = load_and_clean_sheet(excel_path_inicial, 'Funcionalidad oral', clean_cols_funcionalidad_oral)
        df_les_ini = load_and_clean_sheet(excel_path_inicial, 'Lesiones mucosa oral', clean_cols_lesiones_mucosa_oral)
        df_copd_ini = load_and_clean_sheet(excel_path_inicial, 'COPD', clean_cols_copd)
        print("\nTodas las pestañas iniciales cargadas y limpiadas exitosamente.")
        
        # <<< INICIO: Bloque para generar Heatmap de Correlación >>>
        print("\n[PASO ADICIONAL] Generando mapa de calor de correlaciones...")
        
        # Unir los dataframes necesarios que contienen las columnas para el heatmap
        # Merge df_func_ini con df_trasp_ini para obtener edad y sexo
        df_for_heatmap = pd.merge(df_func_ini, df_trasp_ini[['correlativo', 'edad', 'sexo']], on='correlativo', how='left')
        
        # Seleccionar solo las columnas numéricas relevantes que realmente existen
        cols_for_heatmap = ['edad', 'sexo', 
                           'leake_q1', 'leake_q2', 'leake_q3', 'leake_q4', 'leake_q5', 
                           'leake_resultado', 'eichner', 'funcion_masticatoria', 
                           'fuerza_oclusal', 'diadococinesia', 'funcion_deglutoria']
        
        # Verificar qué columnas realmente existen en el DataFrame
        existing_cols = [col for col in cols_for_heatmap if col in df_for_heatmap.columns]
        print(f"  Columnas disponibles para heatmap: {existing_cols}")
        
        if len(existing_cols) > 2:  # Solo generar si hay suficientes columnas
            # Calcular la matriz de correlación
            corr_matrix = df_for_heatmap[existing_cols].corr()
            
            # Generar el gráfico
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, center=0)
            plt.title('Mapa de Calor de Correlación - Variables de Funcionalidad Oral', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Guardar la figura
            heatmap_path = os.path.join(OUTPUT_DIR, 'correlacion_heatmap.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.clf()  # Limpiar la figura para los siguientes gráficos
            print(f"  Mapa de calor guardado en: {heatmap_path}")
        else:
            print("  No hay suficientes columnas numéricas para generar el heatmap")
        # <<< FIN: Bloque para generar Heatmap de Correlación >>>
        
        # --- Feature Engineering (Datos Iniciales) ---
        # BMI
        df_trasp_ini['bmi'] = df_trasp_ini['peso_kg'] / ((df_trasp_ini['altura_cm']/100)**2)
        # Sum caries (asumiendo 2=caries)
        caries_cols_ini = [col for col in df_copd_ini.columns if 'diente_' in col]
        df_copd_ini['sum_caries'] = df_copd_ini[caries_cols_ini].eq(2).sum(axis=1)
        # Has lesion
        df_les_ini['has_lesion'] = (df_les_ini['lesion_elemental'].notna()).astype(int)
        
        # --- Recalcular funcionalidad_oral_final para datos iniciales ---
        df_func_ini['sum_func_components'] = df_func_ini['funcion_masticatoria'] + df_func_ini['fuerza_oclusal'] + df_func_ini['diadococinesia'] + df_func_ini['funcion_deglutoria']
        df_diag_ini['funcionalidad_oral_final'] = (df_func_ini['sum_func_components'] >= 3).astype(int)
        print("  'funcionalidad_oral_final' recalculada para datos iniciales.")
        
        # --- PASO 2: Ejecución de Modelos de Machine Learning (Datos Iniciales) ---
        print("\n[PASO 2/5] Ejecutando Modelos de Machine Learning (Datos Iniciales)...")
        
        # --- Preparación de datos para Modelos V1, V2, V3 ---
        # V1: Solo Funcionalidad oral
        df_v1 = df_func_ini.copy()
        y_v1 = df_v1['funcion_masticatoria'].astype(int)
        cols_to_drop_v1 = ['correlativo', 'leake_resultado', 'eichner', 'funcion_masticatoria', 'sum_func_components']
        X_v1 = df_v1.select_dtypes(include=['float64', 'int64']).drop(columns=[c for c in cols_to_drop_v1 if c in df_v1.select_dtypes(include=['float64', 'int64']).columns])
        
        # V2: Funcionalidad oral + Traspaso
        df_v2_combined = pd.merge(df_func_ini, df_trasp_ini, on='correlativo')
        y_v2 = df_v2_combined['funcion_masticatoria'].astype(int)
        cols_to_drop_v2 = ['correlativo', 'leake_resultado', 'eichner', 'funcion_masticatoria', 'sum_func_components', 'otra_enfermedad_cual', 'farmacos_cuales']
        X_v2 = df_v2_combined.select_dtypes(include=['float64', 'int64']).drop(columns=[c for c in cols_to_drop_v2 if c in df_v2_combined.select_dtypes(include=['float64', 'int64']).columns])
        
        # V3: Funcionalidad oral + Traspaso + Calidad
        df_v3_combined = pd.merge(df_func_ini, df_trasp_ini, on='correlativo')
        df_v3_combined = pd.merge(df_v3_combined, df_cal_ini, on='correlativo')
        y_v3 = df_v3_combined['funcion_masticatoria'].astype(int)
        cols_to_drop_v3 = ['correlativo', 'leake_resultado', 'eichner', 'funcion_masticatoria', 'sum_func_components', 'otra_enfermedad_cual', 'farmacos_cuales']
        X_v3 = df_v3_combined.select_dtypes(include=['float64', 'int64']).drop(columns=[c for c in cols_to_drop_v3 if c in df_v3_combined.select_dtypes(include=['float64', 'int64']).columns])
        
        # --- Ejecución de Modelos V1, V2, V3 con XGBoost ---
        print("\n--- Modelo V1 (Solo Funcionalidad Oral) - XGBoost ---")
        acc_v1_xgb, f1_v1_xgb, auc_v1_xgb, recall_v1_xgb = run_model_pipeline(X_v1, y_v1, 'XGB', 'funcion_masticatoria', 'v1_xgb', use_smote=True)
        
        print("\n--- Modelo V2 (Funcionalidad Oral + Traspaso) - XGBoost ---")
        acc_v2_xgb, f1_v2_xgb, auc_v2_xgb, recall_v2_xgb = run_model_pipeline(X_v2, y_v2, 'XGB', 'funcion_masticatoria', 'v2_xgb', use_smote=True)
        
        print("\n--- Modelo V3 (Funcionalidad Oral + Traspaso + Calidad) - XGBoost ---")
        acc_v3_xgb, f1_v3_xgb, auc_v3_xgb, recall_v3_xgb = run_model_pipeline(X_v3, y_v3, 'XGB', 'funcion_masticatoria', 'v3_xgb', use_smote=True)

        # V4: Todos los datos basales
        df_v4_combined = pd.merge(df_func_ini, df_trasp_ini, on='correlativo')
        df_v4_combined = pd.merge(df_v4_combined, df_cal_ini, on='correlativo')
        df_v4_combined = pd.merge(df_v4_combined, df_les_ini, on='correlativo')
        df_v4_combined = pd.merge(df_v4_combined, df_copd_ini, on='correlativo')
        df_v4_combined = pd.merge(df_v4_combined, df_diag_ini, on='correlativo')
        
        y_v4 = df_v4_combined['funcionalidad_oral_final'].astype(int)
        cols_to_drop_v4 = [
            'correlativo', 'leake_resultado', 'eichner', 'funcion_masticatoria', 'sum_func_components',
            'funcionalidad_oral_final', 'otra_enfermedad_cual', 'farmacos_cuales', 'lesion_nombre_texto'
        ]
        X_v4 = df_v4_combined.select_dtypes(include=['float64', 'int64']).drop(columns=[c for c in cols_to_drop_v4 if c in df_v4_combined.select_dtypes(include=['float64', 'int64']).columns])
        
        # --- Ejecución de Modelos V4 --- (XGB, LR, SVM, RF)
        print("\n--- Modelo V4 (Todos los Datos Basales) ---")
        acc_v4_xgb, f1_v4_xgb, auc_v4_xgb, recall_v4_xgb = run_model_pipeline(X_v4, y_v4, 'XGB', 'funcionalidad_oral_final', 'v4_xgb', use_smote=True)
        acc_v4_lr, f1_v4_lr, auc_v4_lr, recall_v4_lr = run_model_pipeline(X_v4, y_v4, 'LR', 'funcionalidad_oral_final', 'v4_lr', use_smote=True)
        acc_v4_svm, f1_v4_svm, auc_v4_svm, recall_v4_svm = run_model_pipeline(X_v4, y_v4, 'SVM', 'funcionalidad_oral_final', 'v4_svm', use_smote=True)
        acc_v4_rf, f1_v4_rf, auc_v4_rf, recall_v4_rf = run_model_pipeline(X_v4, y_v4, 'RF', 'funcionalidad_oral_final', 'v4_rf', use_smote=True)
        
        # Comparación V4
        print("\n--- Comparación Modelos V4 ---")
        auc_v4_xgb_str = f"{auc_v4_xgb:.4f}" if auc_v4_xgb else "N/A"
        recall_v4_xgb_str = f"{recall_v4_xgb:.4f}" if recall_v4_xgb else "N/A"
        auc_v4_lr_str = f"{auc_v4_lr:.4f}" if auc_v4_lr else "N/A"
        recall_v4_lr_str = f"{recall_v4_lr:.4f}" if recall_v4_lr else "N/A"
        auc_v4_svm_str = f"{auc_v4_svm:.4f}" if auc_v4_svm else "N/A"
        recall_v4_svm_str = f"{recall_v4_svm:.4f}" if recall_v4_svm else "N/A"
        auc_v4_rf_str = f"{auc_v4_rf:.4f}" if auc_v4_rf else "N/A"
        recall_v4_rf_str = f"{recall_v4_rf:.4f}" if recall_v4_rf else "N/A"
        
        print(f"XGB: Acc {acc_v4_xgb:.4f} | F1 {f1_v4_xgb:.4f} | AUC {auc_v4_xgb_str} | Recall Hipo {recall_v4_xgb_str}")
        print(f"LR: Acc {acc_v4_lr:.4f} | F1 {f1_v4_lr:.4f} | AUC {auc_v4_lr_str} | Recall Hipo {recall_v4_lr_str}")
        print(f"SVM: Acc {acc_v4_svm:.4f} | F1 {f1_v4_svm:.4f} | AUC {auc_v4_svm_str} | Recall Hipo {recall_v4_svm_str}")
        print(f"RF: Acc {acc_v4_rf:.4f} | F1 {f1_v4_rf:.4f} | AUC {auc_v4_rf_str} | Recall Hipo {recall_v4_rf_str}")
        
        print("\nModelos de Machine Learning ejecutados exitosamente.")
        
        # --- PASO 3: Cargar datos de seguimiento para comparación ---
        print("\n[PASO 3/5] Cargando datos de seguimiento...")
        df_diag_segu = load_and_clean_sheet(excel_path_seguimiento, 'DIAGNOSTICO', clean_cols_diagnostico)
        df_func_segu = load_and_clean_sheet(excel_path_seguimiento, 'Funcionalidad oral', clean_cols_funcionalidad_oral)
        # <<< INICIO: Carga de datos de Calidad de Vida de Seguimiento >>>
        df_cal_segu = load_and_clean_sheet(excel_path_seguimiento, 'Calidad de vida Basal', clean_cols_calidad_vida)
        # <<< FIN: Carga de datos de Calidad de Vida de Seguimiento >>>

        # Recalcular funcionalidad_oral_final para seguimiento
        df_func_segu['sum_func_components'] = df_func_segu['funcion_masticatoria'] + df_func_segu['fuerza_oclusal'] + df_func_segu['diadococinesia'] + df_func_segu['funcion_deglutoria']
        df_diag_segu['funcionalidad_oral_final'] = (df_func_segu['sum_func_components'] >= 3).astype(int)
        
        # --- PASO 4: Análisis Comparativo ---
        print("\n[PASO 4/5] Ejecutando Análisis Comparativo...")
        
        # Comparación pre/post
        df_comparativo = pd.merge(df_diag_ini[['correlativo', 'funcionalidad_oral_final']], 
                                  df_diag_segu[['correlativo', 'funcionalidad_oral_final']], 
                                  on='correlativo', suffixes=('_ini', '_segu'))
        
        # Cambios func (heatmap)
        cambios = pd.crosstab(df_comparativo['funcionalidad_oral_final_ini'], df_comparativo['funcionalidad_oral_final_segu'], margins=True)
        print("\nCambios Func Oral:\n", cambios)
        sns.heatmap(cambios.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues')
        plt.title('Cambios Func Oral Pre/Post')
        plt.savefig(os.path.join(OUTPUT_DIR, 'cambios_func.png'))
        plt.clf()
        
        # McNemar test
        mask_delta = df_comparativo['funcionalidad_oral_final_ini'].notna() & df_comparativo['funcionalidad_oral_final_segu'].notna()
        if mask_delta.sum() > 1:
            pre = df_comparativo.loc[mask_delta, 'funcionalidad_oral_final_ini'].astype(int)
            post = df_comparativo.loc[mask_delta, 'funcionalidad_oral_final_segu'].astype(int)
            
            table = pd.crosstab(pre, post)
            if table.shape == (2, 2):
                if MCNEMAR_AVAILABLE:
                    result = mcnemar(table, exact=False)
                    print(f"McNemar cambios func: p={result.pvalue:.4f}")
                else:
                    chi2, p_value = mcnemar_simple(table)
                    if p_value is not None:
                        print(f"McNemar cambios func: p={p_value:.4f}")
                    else:
                        print("McNemar: No se pudo calcular")
        
        # <<< INICIO: Bloque de Tests Estadísticos Adicionales >>>
        print("\n--- Tests Estadísticos Adicionales Pre/Post ---")
        
        # 1. Comparación de Calidad de Vida (EQ-5D)
        df_comp_calidad = pd.merge(df_cal_ini[['correlativo', 'eq5d_score_paciente']],
                                   df_cal_segu[['correlativo', 'eq5d_score_paciente']],
                                   on='correlativo', suffixes=('_ini', '_segu'))
        df_comp_calidad.dropna(inplace=True)
        if not df_comp_calidad.empty:
            stat_wilcoxon, p_value_wilcoxon = wilcoxon(df_comp_calidad['eq5d_score_paciente_segu'], df_comp_calidad['eq5d_score_paciente_ini'])
            print(f"\nAnálisis de Calidad de Vida (EQ-5D):")
            print(f"  - Mediana Inicial: {df_comp_calidad['eq5d_score_paciente_ini'].median():.2f}")
            print(f"  - Mediana Seguimiento: {df_comp_calidad['eq5d_score_paciente_segu'].median():.2f}")
            print(f"  - Test de Wilcoxon: p-value = {p_value_wilcoxon:.4f}")
        
        # 2. Comparación de Grado Periodontal
        df_comp_perio = pd.merge(df_diag_ini[['correlativo', 'periodontal_grado']],
                                 df_diag_segu[['correlativo', 'periodontal_grado']],
                                 on='correlativo', suffixes=('_ini', '_segu'))
        df_comp_perio.dropna(inplace=True)
        if not df_comp_perio.empty:
            stat_ttest, p_value_ttest = ttest_rel(df_comp_perio['periodontal_grado_segu'], df_comp_perio['periodontal_grado_ini'])
            print(f"\nAnálisis de Grado Periodontal:")
            print(f"  - Media Inicial: {df_comp_perio['periodontal_grado_ini'].mean():.2f}")
            print(f"  - Media Seguimiento: {df_comp_perio['periodontal_grado_segu'].mean():.2f}")
            print(f"  - Test T pareado: p-value = {p_value_ttest:.4f}")
        # <<< FIN: Bloque de Tests Estadísticos Adicionales >>>
            
        # --- PASO 5: Resumen Final ---
        print("\n[PASO 5/5] Resumen Final de Resultados:")
        print(f"  Modelo V1 (Solo Funcionalidad Oral) - XGB Acc: {acc_v1_xgb:.4f}")
        print(f"  Modelo V2 (Funcionalidad + Traspaso) - XGB Acc: {acc_v2_xgb:.4f}")
        print(f"  Modelo V3 (Funcionalidad + Traspaso + Calidad) - XGB Acc: {acc_v3_xgb:.4f}")
        print(f"  Modelo V4 (Todos los Datos) - XGB Acc: {acc_v4_xgb:.4f}, LR Acc: {acc_v4_lr:.4f}, SVM Acc: {acc_v4_svm:.4f}, RF Acc: {acc_v4_rf:.4f}")
        
        # <<< INICIO: Bloque de Generación de Log de Métricas >>>
        log_filepath = os.path.join(OUTPUT_DIR, "resumen_metricas.txt")
        with open(log_filepath, "w", encoding='utf-8') as f:
            f.write("--- Resumen de Metricas de Modelos V4 ---\n\n")
            f.write(f"Random Forest (RF):\n")
            f.write(f"  - Accuracy: {acc_v4_rf:.4f}\n")
            f.write(f"  - F1-Score (Weighted): {f1_v4_rf:.4f}\n")
            f.write(f"  - AUC: {auc_v4_rf_str}\n")
            f.write(f"  - Recall Hipofuncion: {recall_v4_rf_str}\n\n")
            
            f.write(f"Support Vector Machine (SVM):\n")
            f.write(f"  - Accuracy: {acc_v4_svm:.4f}\n")
            f.write(f"  - F1-Score (Weighted): {f1_v4_svm:.4f}\n")
            f.write(f"  - AUC: {auc_v4_svm_str}\n")
            f.write(f"  - Recall Hipofuncion: {recall_v4_svm_str}\n\n")

            f.write(f"Logistic Regression (LR):\n")
            f.write(f"  - Accuracy: {acc_v4_lr:.4f}\n")
            f.write(f"  - F1-Score (Weighted): {f1_v4_lr:.4f}\n")
            f.write(f"  - AUC: {auc_v4_lr_str}\n")
            f.write(f"  - Recall Hipofuncion: {recall_v4_lr_str}\n\n")

            f.write(f"XGBoost (XGB):\n")
            f.write(f"  - Accuracy: {acc_v4_xgb:.4f}\n")
            f.write(f"  - F1-Score (Weighted): {f1_v4_xgb:.4f}\n")
            f.write(f"  - AUC: {auc_v4_xgb_str}\n")
            f.write(f"  - Recall Hipofuncion: {recall_v4_xgb_str}\n\n")
            
            f.write("\n--- Resumen Tests Estadisticos Pre/Post ---\n\n")
            if 'p_value_wilcoxon' in locals():
                f.write(f"Calidad de Vida (EQ-5D) - Wilcoxon p-value: {p_value_wilcoxon:.4f}\n")
            if 'p_value_ttest' in locals():
                f.write(f"Grado Periodontal - Paired t-test p-value: {p_value_ttest:.4f}\n")

        print(f"\nLog de metricas guardado en: {log_filepath}")
        # <<< FIN: Bloque de Generación de Log de Métricas >>>

    except Exception as e:
        print(f"Ocurrió un error durante el análisis completo: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- ANÁLISIS COMPLETO EDEPAM FINALIZADO ---")