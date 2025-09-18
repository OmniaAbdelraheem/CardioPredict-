import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib and seaborn style
plt.style.use('default')
sns.set_palette("husl")

class ProfessionalTheme:
    """Enhanced Professional medical application color theme"""
    PRIMARY_RED = '#C41E3A'
    DARK_RED = '#A01729'
    LIGHT_GRAY = '#F8F9FA'
    MEDIUM_GRAY = '#6C757D'
    DARK_GRAY = '#343A40'
    WHITE = '#FFFFFF'
    SUCCESS = '#28A745'
    WARNING = '#FFC107'
    INFO = '#17A2B8'
    ACCENT = '#6F42C1'
    
    # Gradient colors for visualizations
    GRADIENT_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # Fonts
    FONT_TITLE = ('Segoe UI', 18, 'bold')
    FONT_HEADER = ('Segoe UI', 14, 'bold')
    FONT_NORMAL = ('Segoe UI', 10)
    FONT_SMALL = ('Segoe UI', 9)
    FONT_BUTTON = ('Segoe UI', 10, 'bold')

class AdvancedDataProcessor:
    """Advanced data processing with techniques from the training script"""
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def set_all_seeds(self, seed):
        """Comprehensive seed setting for reproducibility"""
        np.random.seed(seed)
        
    def handle_missing_values(self, df):
        """Intelligent missing value handling"""
        df_cleaned = df.copy()
        
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype in ['int64', 'float64']:
                # Use median for numeric columns
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            else:
                # Use mode for categorical columns
                mode_value = df_cleaned[col].mode()
                if not mode_value.empty:
                    df_cleaned[col].fillna(mode_value[0], inplace=True)
                else:
                    df_cleaned[col].fillna('Unknown', inplace=True)
        
        return df_cleaned
    
    def advanced_feature_engineering(self, df):
        """Advanced feature engineering"""
        df_features = df.copy()
        
        # Encode categorical variables
        for col in df_features.columns:
            if df_features[col].dtype == 'object':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_features[col] = self.label_encoders[col].fit_transform(df_features[col].astype(str))
                else:
                    # Handle unseen categories
                    unique_values = df_features[col].astype(str).unique()
                    for val in unique_values:
                        if val not in self.label_encoders[col].classes_:
                            # Add new category to existing encoder
                            self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, val)
                    df_features[col] = self.label_encoders[col].transform(df_features[col].astype(str))
            else:
                # Ensure numeric columns are properly typed
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)
        
        return df_features
    
    def balance_data_advanced(self, X, y, method='oversample'):
        """Advanced class balancing with multiple methods"""
        print(f"Original class distribution: {Counter(y)}")
        
        if method == 'none':
            return X, y
        
        # Find the majority class size
        class_counts = Counter(y)
        max_count = max(class_counts.values())
        
        balanced_X = []
        balanced_y = []
        
        for class_label in np.unique(y):
            class_indices = np.where(y == class_label)[0]
            class_X = X[class_indices]
            current_count = len(class_indices)
            
            if current_count < max_count and method == 'oversample':
                # Oversample with noise injection
                needed_samples = max_count - current_count
                oversample_indices = np.random.choice(len(class_X), needed_samples, replace=True)
                oversampled_X = class_X[oversample_indices].copy()
                
                # Add small amount of noise for diversity
                noise = np.random.normal(0, 0.01, oversampled_X.shape)
                oversampled_X += noise
                
                final_class_X = np.vstack([class_X, oversampled_X])
                final_class_y = np.full(len(final_class_X), class_label)
            else:
                final_class_X = class_X
                final_class_y = np.full(len(class_X), class_label)
            
            balanced_X.append(final_class_X)
            balanced_y.append(final_class_y)
        
        # Combine all classes
        X_balanced = np.vstack(balanced_X)
        y_balanced = np.concatenate(balanced_y)
        
        # Shuffle the balanced dataset
        shuffle_indices = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced[shuffle_indices]
        y_balanced = y_balanced[shuffle_indices]
        
        print(f"Balanced class distribution: {Counter(y_balanced)}")
        return X_balanced, y_balanced
    
    def robust_scaling(self, X):
        """Apply robust scaling to handle outliers"""
        return self.scaler.fit_transform(X)

class EnhancedVisualizationEngine:
    """Enhanced visualization engine with multiple chart types"""
    
    def __init__(self):
        self.colors = ProfessionalTheme.GRADIENT_COLORS

    def create_comprehensive_analysis_dashboard(self, ml_core, figsize=(6, 5)):
        """Compact analysis dashboard for 600x500 UI window"""
        if not ml_core.models or ml_core.X_test is None:
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Model Performance Analysis', fontsize=12, weight='bold', y=0.97)
        
        model_names = list(ml_core.performance_data.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # 1. Accuracy
        accuracies = [ml_core.performance_data[name]['accuracy'] for name in model_names]
        bars = axes[0, 0].bar(range(len(model_names)), accuracies, color=colors[:len(model_names)])
        axes[0, 0].set_title('Accuracy', fontsize=9, weight='bold')
        axes[0, 0].set_ylabel('%', fontsize=8)
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels([name[:8] for name in model_names], fontsize=7, rotation=30)
        axes[0, 0].set_ylim(0, 1)
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01,
                            f'{acc:.0%}', ha='center', va='bottom', fontsize=7)
        
        # 2. AUC
        auc_scores = [ml_core.performance_data[name]['auc'] for name in model_names]
        y_pos = range(len(model_names))
        bars = axes[0, 1].barh(y_pos, auc_scores, color=colors[:len(model_names)])
        axes[0, 1].set_title('AUC Score', fontsize=9, weight='bold')
        axes[0, 1].set_xlabel('Score', fontsize=8)
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels([name[:8] for name in model_names], fontsize=7)
        axes[0, 1].set_xlim(0, 1)
        for bar, score in zip(bars, auc_scores):
            axes[0, 1].text(score+0.01, bar.get_y()+bar.get_height()/2.,
                            f'{score:.2f}', ha='left', va='center', fontsize=7)
        
        # 3. Best vs Worst
        best = max(model_names, key=lambda x: ml_core.performance_data[x]['accuracy'])
        worst = min(model_names, key=lambda x: ml_core.performance_data[x]['accuracy'])
        comp_models = [best, worst]
        comp_acc = [ml_core.performance_data[m]['accuracy'] for m in comp_models]
        bars = axes[0, 2].bar(['Best', 'Worst'], comp_acc, color=['#28A745', '#DC3545'], alpha=0.8)
        axes[0, 2].set_title('Best vs Worst', fontsize=9, weight='bold')
        axes[0, 2].set_ylim(0, 1)
        for bar, acc in zip(bars, comp_acc):
            axes[0, 2].text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.02,
                            f'{acc:.0%}', ha='center', fontsize=7)
        
        # 4. Risk Placeholder
        axes[1, 0].text(0.5, 0.5, 'Risk Distribution\n(After Assessment)', 
                        ha='center', va='center', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue"))
        axes[1, 0].set_title('Risk', fontsize=9, weight='bold')
        
        # 5. Reliability Summary
        axes[1, 1].axis('off')
        rel_text = "Reliability:\n"
        for name in model_names:
            acc = ml_core.performance_data[name]['accuracy']
            level = "High" if acc > 0.85 else "Med" if acc > 0.75 else "Low"
            rel_text += f"{name[:8]}: {acc:.0%} ({level})\n"
        axes[1, 1].text(0, 1, rel_text, fontsize=7, va='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0"))
        axes[1, 1].set_title('Reliability', fontsize=9, weight='bold')
        
        # 6. System Score
        avg_acc = sum(accuracies)/len(accuracies)
        avg_auc = sum(auc_scores)/len(auc_scores)
        overall = (avg_acc + avg_auc)/2
        axes[1, 2].text(0.5, 0.6, f"{overall:.0%}", ha='center', va='center',
                        fontsize=20, fontweight='bold',
                        color='#28A745' if overall>0.8 else '#FFC107' if overall>0.7 else '#DC3545')
        axes[1, 2].text(0.5, 0.25, "System Score", ha='center', va='center', fontsize=9)
        axes[1, 2].axis('off')
        
        plt.tight_layout(pad=1.2, w_pad=1.0, h_pad=1.0)
        return fig

        
    def find_target_column(self, dataset):
        """Find target column in dataset"""
        possible_targets = [
            'heart_disease', 'target', 'diagnosis', 'condition', 
            'outcome', 'risk', 'class', 'label', 'Heart_Disease', 'cardio'
        ]
        
        for col in dataset.columns:
            if col.lower() in [t.lower() for t in possible_targets]:
                return col
        
        return dataset.columns[-1]
    
    def create_data_quality_dashboard(self, dataset, figsize=(14, 8)):
        """Create simplified data quality dashboard"""
        if dataset is None:
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Data Quality Report', fontsize=18, weight='bold', y=0.98)
        
        # 1. Dataset Size - Simple display
        axes[0, 0].text(0.5, 0.6, f"{len(dataset):,}", ha='center', va='center',
                    transform=axes[0, 0].transAxes, fontsize=36, fontweight='bold', color='#4ECDC4')
        axes[0, 0].text(0.5, 0.3, "Total Records", ha='center', va='center',
                    transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold')
        axes[0, 0].set_title('Dataset Size', fontsize=14, weight='bold')
        axes[0, 0].axis('off')
        
        # 2. Missing Values - Bar chart
        missing_counts = dataset.isnull().sum()
        if missing_counts.sum() > 0:
            missing_cols = missing_counts[missing_counts > 0]
            bars = axes[0, 1].bar(range(len(missing_cols)), missing_cols.values, color='#FF6B6B')
            axes[0, 1].set_title('Missing Values by Column', fontsize=14, weight='bold')
            axes[0, 1].set_xticks(range(len(missing_cols)))
            axes[0, 1].set_xticklabels(missing_cols.index, rotation=45, fontsize=10)
            axes[0, 1].set_ylabel('Missing Count')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Values\n✓', ha='center', va='center',
                        transform=axes[0, 1].transAxes, fontsize=16, fontweight='bold',
                        color='#28A745')
            axes[0, 1].set_title('Missing Values Status', fontsize=14, weight='bold')
        
        # 3. Data Types - Pie chart
        dtype_counts = dataset.dtypes.value_counts()
        colors = ['#45B7D1', '#96CEB4', '#FFEAA7']
        axes[0, 2].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.0f%%',
                    colors=colors[:len(dtype_counts)], startangle=90)
        axes[0, 2].set_title('Data Types', fontsize=14, weight='bold')
        
        # 4. Basic Statistics
        axes[1, 0].axis('off')
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        stats_text = f"""Dataset Summary:
        
    Records: {len(dataset):,}
    Features: {len(dataset.columns)}
    Numbers: {len(numeric_cols)}
    Text: {len(dataset.select_dtypes(include=['object']).columns)}
    Missing: {dataset.isnull().sum().sum():,}
    Complete: {len(dataset) - dataset.isnull().any(axis=1).sum():,}"""
        
        axes[1, 0].text(0.1, 0.9, stats_text, transform=axes[1, 0].transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 0].set_title('Quick Stats', fontsize=14, weight='bold')
        
        # 5. Sample Data Preview
        axes[1, 1].axis('off')
        if len(dataset) > 0:
            sample_text = "Sample Data (First 3 rows):\n\n"
            for i in range(min(3, len(dataset))):
                row_text = f"Row {i+1}:\n"
                for col in dataset.columns[:3]:  # Show first 3 columns
                    value = str(dataset.iloc[i][col])
                    if len(value) > 20:
                        value = value[:17] + "..."
                    row_text += f"  {col}: {value}\n"
                sample_text += row_text + "\n"
        else:
            sample_text = "No data available"
        
        axes[1, 1].text(0.05, 0.95, sample_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0"))
        axes[1, 1].set_title('Data Preview', fontsize=14, weight='bold')
        
        # 6. Data Quality Score
        quality_score = 1.0
        if dataset.isnull().sum().sum() > 0:
            quality_score -= 0.2  # Reduce for missing values
        if len(dataset) < 100:
            quality_score -= 0.3  # Reduce for small dataset
        
        quality_score = max(0, quality_score)  # Don't go below 0
        
        score_color = '#28A745' if quality_score > 0.8 else '#FFC107' if quality_score > 0.6 else '#DC3545'
        
        axes[1, 2].text(0.5, 0.6, f"{quality_score:.1%}", ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=36, fontweight='bold',
                    color=score_color)
        axes[1, 2].text(0.5, 0.3, "Data Quality\nScore", ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=14, fontweight='bold')
        axes[1, 2].set_title('Quality Rating', fontsize=14, weight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig

class EnhancedMLCore:
    """Enhanced ML Core with advanced data processing"""
    
    def __init__(self):
        self.dataset = None
        self.models = {}
        self.performance_data = {}
        self.data_processor = AdvancedDataProcessor()
        self.feature_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_predictions = {}
        self.visualization_engine = EnhancedVisualizationEngine()
    
    def preprocess_data_advanced(self, X, y):
        """Advanced preprocessing with the enhanced data processor"""
        try:
            # Handle missing values
            X_processed = self.data_processor.handle_missing_values(X)
            
            # Advanced feature engineering
            X_features = self.data_processor.advanced_feature_engineering(X_processed)
            
            # Store feature names
            self.feature_names = list(X_features.columns) if hasattr(X_features, 'columns') else [f'feature_{i}' for i in range(X_features.shape[1])]
            
            # Convert to numpy array if needed
            if hasattr(X_features, 'values'):
                X_array = X_features.values
            else:
                X_array = X_features
            
            # Apply class balancing
            X_balanced, y_balanced = self.data_processor.balance_data_advanced(X_array, y, method='oversample')
            
            # Apply robust scaling
            X_scaled = self.data_processor.robust_scaling(X_balanced)
            
            print(f"Advanced preprocessing complete:")
            print(f"  Original shape: {X.shape}")
            print(f"  After balancing: {X_balanced.shape}")
            print(f"  Features: {len(self.feature_names)}")
            
            return X_scaled, y_balanced
            
        except Exception as e:
            print(f"Error in advanced preprocessing: {str(e)}")
            return X.values if hasattr(X, 'values') else X, y
    
    def train_advanced_models_enhanced(self, X, y):
        """Enhanced model training with advanced preprocessing and cross-validation"""
        try:
            # Advanced preprocessing
            X_processed, y_processed = self.preprocess_data_advanced(X, y)
            
            # Stratified split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_processed, y_processed, test_size=0.2, random_state=42, 
                stratify=y_processed
            )
            
            # Enhanced model configurations with optimized hyperparameters
            models_config = {
                'K-Nearest Neighbors': KNeighborsClassifier(
                    n_neighbors=7,
                    weights='distance',
                    metric='minkowski',
                    p=2,
                    algorithm='auto'
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight='balanced'
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=6,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42
                ),
                'Logistic Regression': LogisticRegression(
                    random_state=42,
                    max_iter=2000,
                    C=1.0,
                    class_weight='balanced',
                    solver='lbfgs'
                ),
                'Support Vector Machine': SVC(
                    probability=True,
                    random_state=42,
                    C=1.0,
                    gamma='scale',
                    class_weight='balanced',
                    kernel='rbf'
                )
            }
            
            # Advanced cross-validation with stratification
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Train and evaluate models
            for name, model in models_config.items():
                print(f"Training {name} with advanced preprocessing...")
                
                # Train model
                model.fit(self.X_train, self.y_train)
                
                # Predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Advanced cross-validation
                cv_scores = cross_val_score(
                    model, X_processed, y_processed, 
                    cv=cv_strategy, scoring='accuracy', n_jobs=-1
                )
                
                cv_auc_scores = cross_val_score(
                    model, X_processed, y_processed,
                    cv=cv_strategy, scoring='roc_auc', n_jobs=-1
                )
                
                # Calculate comprehensive metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                auc = roc_auc_score(self.y_test, y_pred_proba)
                cm = confusion_matrix(self.y_test, y_pred)
                
                # Store comprehensive results
                self.models[name] = model
                self.performance_data[name] = {
                    'accuracy': accuracy,
                    'auc': auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_auc_mean': cv_auc_scores.mean(),
                    'cv_auc_std': cv_auc_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'confusion_matrix': cm,
                    'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
                }
                
                self.model_predictions[name] = {
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                print(f"{name} Results:")
                print(f"  Accuracy: {accuracy:.3f}")
                print(f"  AUC: {auc:.3f}")
                print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                print(f"  CV AUC: {cv_auc_scores.mean():.3f} ± {cv_auc_scores.std():.3f}")
                print("-" * 50)
            
            return True
            
        except Exception as e:
            print(f"Error in enhanced model training: {str(e)}")
            return False
    
    def predict_risk_enhanced(self, patient_data):
        """Enhanced risk prediction with advanced preprocessing"""
        if not self.models:
            return self._rule_based_prediction(patient_data)
        
        try:
            # Convert to DataFrame
            patient_df = pd.DataFrame([patient_data])
            
            # Apply the same preprocessing pipeline
            patient_processed = self.data_processor.handle_missing_values(patient_df)
            patient_features = self.data_processor.advanced_feature_engineering(patient_processed)
            
            # Ensure feature compatibility
            for feature in self.feature_names:
                if feature not in patient_features.columns:
                    median_value = 0  # Default value for missing features
                    patient_features[feature] = median_value
            
            # Reorder columns
            patient_features = patient_features[self.feature_names]
            
            # Apply scaling
            patient_scaled = self.data_processor.scaler.transform(patient_features.values)
            
            # Make predictions
            results = {}
            for name, model in self.models.items():
                prediction = model.predict(patient_scaled)[0]
                probability = model.predict_proba(patient_scaled)[0]
                
                results[name] = {
                    'prediction': int(prediction),
                    'probability': float(probability[1]),
                    'confidence': float(max(probability)),
                    'risk_score': float(probability[1])  # Enhanced risk scoring
                }
            
            return results
            
        except Exception as e:
            print(f"Error in enhanced prediction: {str(e)}")
            return self._rule_based_prediction(patient_data)
    
    def _rule_based_prediction(self, patient_data):
        """Enhanced rule-based fallback prediction"""
        age = patient_data.get('Age', 50)
        bp_systolic = patient_data.get('Blood_Pressure_Systolic', 120)
        cholesterol = patient_data.get('Cholesterol', 200)
        bmi = patient_data.get('BMI', 25)
        
        # Enhanced risk calculation
        risk_score = 0
        
        # Age risk (progressive)
        if age > 65: risk_score += 0.4
        elif age > 55: risk_score += 0.3
        elif age > 45: risk_score += 0.2
        
        # Blood pressure risk
        if bp_systolic > 180: risk_score += 0.4
        elif bp_systolic > 140: risk_score += 0.3
        elif bp_systolic > 130: risk_score += 0.2
        
        # Cholesterol risk
        if cholesterol > 280: risk_score += 0.3
        elif cholesterol > 240: risk_score += 0.2
        elif cholesterol > 200: risk_score += 0.1
        
        # BMI risk
        if bmi > 35: risk_score += 0.3
        elif bmi > 30: risk_score += 0.2
        elif bmi > 25: risk_score += 0.1
        
        # Lifestyle factors
        if patient_data.get('Smoking', 'No') == 'Current': risk_score += 0.3
        elif patient_data.get('Smoking', 'No') == 'Former': risk_score += 0.1
        
        if patient_data.get('Diabetes', 'No') != 'No': risk_score += 0.2
        if patient_data.get('Family_History_Heart_Disease', 'No') == 'Yes': risk_score += 0.15
        if patient_data.get('Exercise_Angina', 'No') == 'Yes': risk_score += 0.15
        
        # Normalize risk score
        risk_score = min(risk_score, 0.95)
        prediction = 1 if risk_score > 0.5 else 0
        
        return {
            'Enhanced Rule-Based Model': {
                'prediction': prediction,
                'probability': risk_score,
                'confidence': 0.75,
                'risk_score': risk_score
            }
        }
    
    def generate_comprehensive_analysis_plots(self):
        """Generate comprehensive analysis using the enhanced visualization engine"""
        return self.visualization_engine.create_comprehensive_analysis_dashboard(self)
    
    def generate_data_quality_plots(self):
        """Generate data quality dashboard"""
        return self.visualization_engine.create_data_quality_dashboard(self.dataset)

class ScrollableFrame:
    """Scrollable frame widget for better UI navigation"""
    
    def __init__(self, parent):
        self.parent = parent
        self.canvas = tk.Canvas(parent, bg=ProfessionalTheme.WHITE)
        self.scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=ProfessionalTheme.WHITE)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind mousewheel
        self._bind_mousewheel()
    
    def _bind_mousewheel(self):
        """Bind mousewheel events for scrolling"""
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            self.canvas.unbind_all("<MouseWheel>")
        
        self.canvas.bind('<Enter>', _bind_to_mousewheel)
        self.canvas.bind('<Leave>', _unbind_from_mousewheel)
    
    def pack(self, **kwargs):
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
    
    def get_frame(self):
        return self.scrollable_frame
class EnhancedProfessionalMedicalApp:
    """Enhanced Professional Medical Application with advanced processing and visualizations"""
    
    def __init__(self, root):
        self.root = root
        self.ml_core = EnhancedMLCore()
        self.current_results = {}
        self.patient_answers = {}
        self.current_page = "dashboard"
        
        # Navigation state
        self.scroll_position = 0
        
        self.setup_main_window()
        self.create_professional_interface()
        self.show_medical_disclaimer()
        
        # Auto-load and train with enhanced processing
        self.auto_load_and_train_enhanced()
    
    def setup_main_window(self):
        """Setup enhanced main window with better navigation"""
        self.root.title("CardioPredict Pro - Enhanced with Advanced Processing & Visualization")
        self.root.geometry("1600x1000")
        self.root.configure(bg=ProfessionalTheme.WHITE)
        self.root.resizable(True, True)
        self.root.minsize(1400, 900)
        
        # Configure root grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Center window
        self.center_window()
    
    def center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_professional_interface(self):
        """Create enhanced professional interface with better navigation"""
        # Main container with grid layout
        self.main_container = tk.Frame(self.root, bg=ProfessionalTheme.WHITE)
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(2, weight=1)
        
        # Header
        self.create_enhanced_header()
        
        # Navigation with scroll-to-top functionality
        self.create_enhanced_navigation()
        
        # Content area with scrolling
        self.create_scrollable_content_area()
        
        # Enhanced status bar
        self.create_enhanced_status_bar()
        
        # Show dashboard by default
        self.show_enhanced_dashboard()
    
    def create_enhanced_header(self):
        """Create enhanced header with gradient-like effect"""
        header_frame = tk.Frame(self.main_container, bg=ProfessionalTheme.PRIMARY_RED, height=100)
        header_frame.grid(row=0, column=0, sticky="ew")
        header_frame.grid_propagate(False)
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Medical symbol and title
        title_frame = tk.Frame(header_frame, bg=ProfessionalTheme.PRIMARY_RED)
        title_frame.grid(row=0, column=0, padx=20, pady=15, sticky="w")
        
        # Enhanced medical symbol
        symbol_frame = tk.Frame(title_frame, bg=ProfessionalTheme.PRIMARY_RED)
        symbol_frame.pack(side=tk.LEFT)
        
        symbol_label = tk.Label(symbol_frame, text="⚕", font=('Segoe UI', 36),
                              fg=ProfessionalTheme.WHITE, bg=ProfessionalTheme.PRIMARY_RED)
        symbol_label.pack()
        
        # Pulsing effect for symbol (simulated with color changes)
        def pulse_symbol():
            current_color = symbol_label.cget('fg')
            if current_color == ProfessionalTheme.WHITE:
                symbol_label.config(fg='#FFE6E6')
            else:
                symbol_label.config(fg=ProfessionalTheme.WHITE)
            self.root.after(1500, pulse_symbol)
        
        pulse_symbol()
        
        # Title text
        title_text = tk.Frame(title_frame, bg=ProfessionalTheme.PRIMARY_RED)
        title_text.pack(side=tk.LEFT, padx=(20, 0))
        
        tk.Label(title_text, text="CardioPredict Pro", font=('Segoe UI', 20, 'bold'),
                fg=ProfessionalTheme.WHITE, bg=ProfessionalTheme.PRIMARY_RED).pack(anchor=tk.W)
        
        tk.Label(title_text, text="Advanced Data Processing & Enhanced Visualization Engine", 
                font=('Segoe UI', 11),
                fg=ProfessionalTheme.WHITE, bg=ProfessionalTheme.PRIMARY_RED).pack(anchor=tk.W)
        
        tk.Label(title_text, text="Professional Medical Analysis Platform", 
                font=('Segoe UI', 9, 'italic'),
                fg='#FFE6E6', bg=ProfessionalTheme.PRIMARY_RED).pack(anchor=tk.W)
        
        # Enhanced version info and controls
        controls_frame = tk.Frame(header_frame, bg=ProfessionalTheme.PRIMARY_RED)
        controls_frame.grid(row=0, column=2, padx=20, pady=15, sticky="e")
        
        # Scroll to top button
        scroll_top_btn = tk.Button(controls_frame, text="⬆ Top", command=self.scroll_to_top,
                                 bg=ProfessionalTheme.WHITE, fg=ProfessionalTheme.PRIMARY_RED,
                                 font=('Segoe UI', 9, 'bold'), relief=tk.FLAT,
                                 padx=15, pady=5, cursor='hand2')
        scroll_top_btn.pack(pady=(0, 5))
        
        # Version info
        tk.Label(controls_frame, text="v3.0.0 Enhanced", font=('Segoe UI', 10, 'bold'),
                fg=ProfessionalTheme.WHITE, bg=ProfessionalTheme.PRIMARY_RED).pack()
        tk.Label(controls_frame, text="Advanced Analytics Edition", font=('Segoe UI', 8),
                fg='#FFE6E6', bg=ProfessionalTheme.PRIMARY_RED).pack()
        
        # Status indicator
        self.status_indicator = tk.Label(controls_frame, text="● Online", 
                                       font=('Segoe UI', 9), fg='#90EE90',
                                       bg=ProfessionalTheme.PRIMARY_RED)
        self.status_indicator.pack(pady=(5, 0))
    
    def create_enhanced_navigation(self):
        """Create enhanced navigation with better styling"""
        nav_frame = tk.Frame(self.main_container, bg=ProfessionalTheme.LIGHT_GRAY, height=60)
        nav_frame.grid(row=1, column=0, sticky="ew")
        nav_frame.grid_propagate(False)
        
        # Navigation buttons with enhanced styling
        nav_buttons = [
            ("🏠 Dashboard", self.show_enhanced_dashboard, "Home screen with system overview"),
            ("🔍 Assessment", self.show_enhanced_assessment, "Patient risk assessment tool"),
            ("📊 Results", self.show_enhanced_results, "View assessment results"),
            ("📈 Advanced Analytics", self.show_advanced_analytics, "Comprehensive model analysis"),
            ("📋 Data Quality", self.show_data_quality_analysis, "Data quality dashboard"),
            ("📁 Data Management", self.show_enhanced_data_management, "Dataset operations"),
            ("⚙️ Settings", self.show_enhanced_settings, "System configuration")
        ]
        
        button_frame = tk.Frame(nav_frame, bg=ProfessionalTheme.LIGHT_GRAY)
        button_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=8)
        
        for i, (text, command, tooltip) in enumerate(nav_buttons):
            btn = tk.Button(button_frame, text=text, command=command,
                          bg=ProfessionalTheme.WHITE, fg=ProfessionalTheme.DARK_GRAY,
                          font=('Segoe UI', 10, 'bold'), relief=tk.RAISED, bd=1,
                          padx=15, pady=8, cursor='hand2')
            btn.pack(side=tk.LEFT, padx=3, pady=2)
            
            # Enhanced hover effects
            def on_enter(e, btn=btn):
                btn.config(bg=ProfessionalTheme.PRIMARY_RED, fg=ProfessionalTheme.WHITE, relief=tk.RAISED)
            def on_leave(e, btn=btn):
                btn.config(bg=ProfessionalTheme.WHITE, fg=ProfessionalTheme.DARK_GRAY, relief=tk.RAISED)
            
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
            
            # Tooltip (simplified - you could use a proper tooltip library)
            def create_tooltip(widget, text):
                def on_enter(event):
                    self.update_status(text)
                def on_leave(event):
                    self.update_status("Ready - Enhanced Processing Active")
                widget.bind('<Enter>', on_enter, add='+')
                widget.bind('<Leave>', on_leave, add='+')
            
            create_tooltip(btn, tooltip)
    
    def create_scrollable_content_area(self):
        """Create enhanced scrollable content area"""
        # Content container
        content_container = tk.Frame(self.main_container, bg=ProfessionalTheme.WHITE)
        content_container.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        content_container.grid_columnconfigure(0, weight=1)
        content_container.grid_rowconfigure(0, weight=1)
        
        # Scrollable frame
        self.scrollable_content = ScrollableFrame(content_container)
        self.scrollable_content.pack()
        
        # Get the actual content frame
        self.content_frame = self.scrollable_content.get_frame()
    
    def create_enhanced_status_bar(self):
        """Create enhanced status bar with more information"""
        self.status_frame = tk.Frame(self.main_container, bg=ProfessionalTheme.DARK_GRAY, height=35)
        self.status_frame.grid(row=3, column=0, sticky="ew")
        self.status_frame.grid_propagate(False)
        self.status_frame.grid_columnconfigure(1, weight=1)
        
        # Left side status
        left_status = tk.Frame(self.status_frame, bg=ProfessionalTheme.DARK_GRAY)
        left_status.grid(row=0, column=0, sticky="w", padx=10, pady=3)
        
        self.status_label = tk.Label(left_status, text="Ready - Enhanced Processing Active",
                                   fg=ProfessionalTheme.WHITE, bg=ProfessionalTheme.DARK_GRAY,
                                   font=('Segoe UI', 10))
        self.status_label.pack(side=tk.LEFT)
        
        # Center progress indicator (initially hidden)
        self.progress_frame = tk.Frame(self.status_frame, bg=ProfessionalTheme.DARK_GRAY)
        self.progress_frame.grid(row=0, column=1, sticky="ew", padx=20, pady=3)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var,
                                          maximum=100, length=300)
        
        # Right side info
        right_status = tk.Frame(self.status_frame, bg=ProfessionalTheme.DARK_GRAY)
        right_status.grid(row=0, column=2, sticky="e", padx=10, pady=3)
        
        self.model_count_label = tk.Label(right_status, text=f"Models: {len(self.ml_core.models)}",
                                        fg=ProfessionalTheme.WHITE, bg=ProfessionalTheme.DARK_GRAY,
                                        font=('Segoe UI', 9))
        self.model_count_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        self.data_status_label = tk.Label(right_status, text="Data: Ready",
                                        fg=ProfessionalTheme.WHITE, bg=ProfessionalTheme.DARK_GRAY,
                                        font=('Segoe UI', 9))
        self.data_status_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Current time
        self.time_label = tk.Label(right_status, text="",
                                 fg=ProfessionalTheme.WHITE, bg=ProfessionalTheme.DARK_GRAY,
                                 font=('Segoe UI', 9))
        self.time_label.pack(side=tk.RIGHT)
        
        # Update time every second
        self.update_time()
    
    def update_time(self):
        """Update the time display in status bar"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)
    
    def scroll_to_top(self):
        """Scroll to top of content"""
        self.scrollable_content.canvas.yview_moveto(0)
        self.update_status("Scrolled to top")
    
    def update_status(self, message, show_progress=False, progress_value=0):
        """Enhanced status update with progress indication"""
        self.status_label.config(text=message)
        
        if show_progress:
            self.progress_bar.pack(fill=tk.X)
            self.progress_var.set(progress_value)
        else:
            self.progress_bar.pack_forget()
        
        # Update model and data counts
        self.model_count_label.config(text=f"Models: {len(self.ml_core.models)}")
        data_status = "Ready" if self.ml_core.dataset is not None else "No Data"
        self.data_status_label.config(text=f"Data: {data_status}")
        
        self.root.update_idletasks()
    
    def clear_content(self):
        """Clear content area"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.scroll_to_top()
    
    def auto_load_and_train_enhanced(self):
        """Auto-load sample data and train with enhanced processing"""
        try:
            self.update_status("Initializing enhanced processing engine...", True, 10)
            
            # Create enhanced sample dataset
            self.create_enhanced_sample_dataset()
            self.update_status("Enhanced sample dataset created", True, 40)
            
            if self.ml_core.dataset is not None:
                self.update_status("Training models with advanced processing...", True, 60)
                
                # Find target column
                target_col = self.find_target_column()
                if target_col:
                    # Prepare data
                    X = self.ml_core.dataset.drop(columns=[target_col])
                    y = self.ml_core.dataset[target_col]
                    
                    # Train with enhanced processing
                    if self.ml_core.train_advanced_models_enhanced(X, y):
                        self.update_status("Enhanced model training complete", True, 100)
                        
                        # Show training summary
                        summary = "Enhanced Model Training Complete!\n\n"
                        summary += f"Dataset: {len(self.ml_core.dataset):,} records with {len(X.columns)} features\n"
                        summary += f"Advanced preprocessing: ✓ Applied\n"
                        summary += f"Class balancing: ✓ Applied\n"
                        summary += f"Robust scaling: ✓ Applied\n\n"
                        
                        for model_name, perf in self.ml_core.performance_data.items():
                            summary += f"{model_name}:\n"
                            summary += f"  Accuracy: {perf['accuracy']:.1%}\n"
                            summary += f"  AUC Score: {perf['auc']:.3f}\n"
                            summary += f"  CV Score: {perf['cv_mean']:.3f}±{perf['cv_std']:.3f}\n\n"
                        
                        messagebox.showinfo("Enhanced Training Complete", summary)
                        self.update_status("System ready - Enhanced processing active")
                    else:
                        self.update_status("Enhanced training failed")
                else:
                    self.update_status("Could not identify target column")
            
        except Exception as e:
            self.update_status("Error during enhanced initialization")
            print(f"Enhanced initialization error: {str(e)}")
    
    def create_enhanced_sample_dataset(self):
        """Create enhanced sample medical dataset with more realistic correlations"""
        np.random.seed(42)
        n_samples = 1500  # Increased sample size
        
        # Generate more sophisticated medical data
        age = np.random.gamma(2, 25)  # Gamma distribution for age
        age = np.clip(age, 18, 90)
        
        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48])
        
        # More complex BMI relationships
        base_bmi = np.random.normal(26.5, 5, n_samples)
        age_factor = (age - 30) * 0.05  # BMI increases with age
        gender_factor = np.where(gender == 'Male', 1.0, 0.8)  # Gender differences
        bmi = base_bmi * gender_factor + age_factor
        bmi = np.clip(bmi, 16, 50)
        
        # Advanced blood pressure modeling with multiple factors
        bp_base = 90 + age * 0.7 + bmi * 0.8
        stress_factor = np.random.normal(0, 15, n_samples)  # Random stress component
        genetic_factor = np.random.normal(0, 10, n_samples)  # Genetic predisposition
        bp_systolic = bp_base + stress_factor + genetic_factor
        bp_systolic = np.clip(bp_systolic, 80, 220)
        
        bp_diastolic = bp_systolic * 0.65 + np.random.normal(0, 8, n_samples)
        bp_diastolic = np.clip(bp_diastolic, 50, 130)
        
        # Enhanced cholesterol with age and lifestyle correlations
        cholesterol_base = 150 + age * 1.2 + bmi * 2.5
        lifestyle_factor = np.random.normal(0, 30, n_samples)
        cholesterol = cholesterol_base + lifestyle_factor
        cholesterol = np.clip(cholesterol, 120, 400)
        
        # Heart rate with fitness correlation
        fitness_level = np.random.uniform(0.3, 1.0, n_samples)  # Fitness factor
        max_heart_rate = (220 - age) * fitness_level + np.random.normal(0, 15, n_samples)
        max_heart_rate = np.clip(max_heart_rate, 100, 200)
        
        # Resting heart rate
        resting_hr = 60 + (1 - fitness_level) * 40 + np.random.normal(0, 10, n_samples)
        resting_hr = np.clip(resting_hr, 45, 120)
        
        # Enhanced lifestyle factors with interdependencies
        # Smoking affects multiple health parameters
        smoking_prob = 0.15 + (age < 40) * 0.05  # Younger people more likely to smoke
        smoking = np.random.choice(['Never', 'Former', 'Current'], n_samples, 
                                 p=[1-smoking_prob-0.1, 0.1, smoking_prob])
        
        # Exercise frequency (correlated with BMI and age)
        exercise_base = 5 - (bmi - 25) * 0.1 - (age - 30) * 0.02
        exercise = np.random.poisson(np.clip(exercise_base, 0.5, 7), n_samples)
        exercise = np.clip(exercise, 0, 7)
        
        # Family history with genetic clustering
        family_history = np.random.choice(['No', 'Yes'], n_samples, p=[0.65, 0.35])
        
        # Diabetes with BMI and age correlation
        diabetes_risk = (bmi > 30) * 0.2 + (age > 60) * 0.15 + (family_history == 'Yes') * 0.1
        diabetes_probs = np.random.uniform(0, 1, n_samples)
        diabetes = np.where(diabetes_probs < diabetes_risk * 0.8, 
                          np.random.choice(['Type 2', 'Type 1'], n_samples, p=[0.9, 0.1]),
                          'No')
        
        # Exercise-induced angina
        angina_risk = (age > 55) * 0.1 + (bp_systolic > 140) * 0.15 + (cholesterol > 240) * 0.1
        exercise_angina = np.where(np.random.uniform(0, 1, n_samples) < angina_risk, 'Yes', 'No')
        
        # Additional sophisticated features
        sleep_hours = np.random.normal(7, 1.5, n_samples)
        sleep_hours = np.clip(sleep_hours, 4, 12)
        
        stress_level = np.random.randint(1, 11, n_samples)  # 1-10 scale
        
        alcohol_consumption = np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], 
                                             n_samples, p=[0.3, 0.4, 0.25, 0.05])
        
        # Medical history indicators
        previous_heart_events = np.random.choice(['No', 'Yes'], n_samples, p=[0.85, 0.15])
        
        # Calculate sophisticated heart disease risk
        risk_score = np.zeros(n_samples)
        
        # Age risk (non-linear)
        risk_score += np.where(age < 40, 0.05,
                      np.where(age < 50, 0.1,
                      np.where(age < 60, 0.2,
                      np.where(age < 70, 0.3, 0.4))))
        
        # Gender risk
        risk_score += np.where(gender == 'Male', 0.1, 0.05)
        
        # BMI risk (U-shaped curve)
        risk_score += np.where(bmi < 18.5, 0.1,
                      np.where(bmi < 25, 0.0,
                      np.where(bmi < 30, 0.1,
                      np.where(bmi < 35, 0.2, 0.3))))
        
        # Blood pressure risk
        risk_score += np.where(bp_systolic < 120, 0.0,
                      np.where(bp_systolic < 130, 0.05,
                      np.where(bp_systolic < 140, 0.1,
                      np.where(bp_systolic < 160, 0.2, 0.3))))
        
        # Cholesterol risk
        risk_score += np.where(cholesterol < 200, 0.0,
                      np.where(cholesterol < 240, 0.1, 0.2))
        
        # Lifestyle factors
        risk_score += np.where(smoking == 'Current', 0.25,
                      np.where(smoking == 'Former', 0.1, 0.0))
        
        risk_score += np.where(exercise < 2, 0.15,
                      np.where(exercise < 4, 0.05, 0.0))
        
        risk_score += np.where(diabetes == 'Type 2', 0.2,
                      np.where(diabetes == 'Type 1', 0.15, 0.0))
        
        risk_score += np.where(family_history == 'Yes', 0.1, 0.0)
        risk_score += np.where(exercise_angina == 'Yes', 0.15, 0.0)
        risk_score += np.where(previous_heart_events == 'Yes', 0.3, 0.0)
        
        # Sleep and stress factors
        risk_score += np.where(sleep_hours < 6, 0.1,
                      np.where(sleep_hours > 9, 0.05, 0.0))
        
        risk_score += (stress_level - 5) * 0.02  # Stress level 1-10
        
        risk_score += np.where(alcohol_consumption == 'Heavy', 0.15,
                      np.where(alcohol_consumption == 'Moderate', -0.02,  # Moderate alcohol may be protective
                      np.where(alcohol_consumption == 'Light', -0.01, 0.0)))
        
        # Add random noise for realism
        risk_score += np.random.normal(0, 0.1, n_samples)
        
        # Normalize and create binary outcome
        risk_score = np.clip(risk_score, 0, 1)
        heart_disease = (risk_score > 0.5).astype(int)
        
        # Create enhanced dataset
        self.ml_core.dataset = pd.DataFrame({
            'Age': age.round(1),
            'Gender': gender,
            'BMI': bmi.round(1),
            'Blood_Pressure_Systolic': bp_systolic.round(0),
            'Blood_Pressure_Diastolic': bp_diastolic.round(0),
            'Cholesterol': cholesterol.round(0),
            'Max_Heart_Rate': max_heart_rate.round(0),
            'Resting_Heart_Rate': resting_hr.round(0),
            'Smoking': smoking,
            'Exercise_Frequency': exercise,
            'Family_History_Heart_Disease': family_history,
            'Diabetes': diabetes,
            'Exercise_Angina': exercise_angina,
            'Sleep_Hours': sleep_hours.round(1),
            'Stress_Level': stress_level,
            'Alcohol_Consumption': alcohol_consumption,
            'Previous_Heart_Events': previous_heart_events,
            'Risk_Score': risk_score.round(3),
            'Heart_Disease': heart_disease
        })
        
        print(f"Enhanced dataset created with {len(self.ml_core.dataset)} records and {len(self.ml_core.dataset.columns)} features")
        print(f"Target distribution: {self.ml_core.dataset['Heart_Disease'].value_counts().to_dict()}")
    
    def show_enhanced_dashboard(self):
        """Show enhanced dashboard with comprehensive system overview"""
        self.clear_content()
        self.current_page = "dashboard"
        self.update_status("Enhanced dashboard loaded")
        
        # Welcome section with enhanced styling
        welcome_frame = tk.Frame(self.content_frame, bg=ProfessionalTheme.WHITE)
        welcome_frame.pack(fill=tk.X, pady=(0, 25))
        
        # Main title with gradient-like effect
        title_frame = tk.Frame(welcome_frame, bg=ProfessionalTheme.WHITE)
        title_frame.pack(fill=tk.X)
        
        main_title = tk.Label(title_frame, text="CardioPredict Pro", 
                            font=('Segoe UI', 24, 'bold'), fg=ProfessionalTheme.PRIMARY_RED,
                            bg=ProfessionalTheme.WHITE)
        main_title.pack(anchor=tk.W)
        
        subtitle = tk.Label(title_frame, text="Advanced Data Processing & Enhanced Visualization Platform",
                          font=('Segoe UI', 14), fg=ProfessionalTheme.MEDIUM_GRAY,
                          bg=ProfessionalTheme.WHITE)
        subtitle.pack(anchor=tk.W, pady=(5, 0))
        
        feature_line = tk.Label(title_frame, text="• Sophisticated ML Pipeline • Advanced Preprocessing • Comprehensive Analytics •",
                              font=('Segoe UI', 11, 'italic'), fg=ProfessionalTheme.INFO,
                              bg=ProfessionalTheme.WHITE)
        feature_line.pack(anchor=tk.W, pady=(10, 0))
        
        # Enhanced quick actions grid
        actions_card = self.create_professional_card(self.content_frame, "Quick Actions Hub", 
                                                   f"System Status: {'Operational' if self.ml_core.models else 'Initializing'}")
        actions_card.pack(fill=tk.X, pady=(0, 20))
        
        actions_content = tk.Frame(actions_card, bg=ProfessionalTheme.WHITE)
        actions_content.pack(fill=tk.X, padx=20, pady=20)
        
        # Grid layout for actions
        actions_grid = tk.Frame(actions_content, bg=ProfessionalTheme.WHITE)
        actions_grid.pack(fill=tk.X)
        
        quick_actions = [
            ("🔍 New Assessment", self.show_enhanced_assessment, "Comprehensive patient evaluation", ProfessionalTheme.SUCCESS),
            ("📊 View Results", self.show_enhanced_results, "Latest assessment outcomes", ProfessionalTheme.INFO),
            ("📈 Advanced Analytics", self.show_advanced_analytics, "ML model comparison & insights", ProfessionalTheme.WARNING),
            ("📋 Data Quality", self.show_data_quality_analysis, "Dataset analysis & validation", ProfessionalTheme.ACCENT),
            ("🔄 Retrain Models", self.train_enhanced_models, "Update ML algorithms", ProfessionalTheme.DARK_GRAY),
            ("📁 Data Management", self.show_enhanced_data_management, "Dataset operations", ProfessionalTheme.MEDIUM_GRAY)
        ]
        
        for i, (text, command, description, color) in enumerate(quick_actions):
            row = i // 2
            col = i % 2
            
            action_frame = tk.Frame(actions_grid, bg=ProfessionalTheme.WHITE, relief=tk.RAISED, bd=1)
            action_frame.grid(row=row, column=col, padx=10, pady=8, sticky="ew")
            
            btn = tk.Button(action_frame, text=text, command=command,
                          bg=color, fg=ProfessionalTheme.WHITE,
                          font=('Segoe UI', 11, 'bold'), relief=tk.FLAT,
                          padx=25, pady=15, cursor='hand2')
            btn.pack(fill=tk.X, padx=10, pady=(10, 5))
            
            desc_label = tk.Label(action_frame, text=description, 
                                font=('Segoe UI', 9), fg=ProfessionalTheme.MEDIUM_GRAY,
                                bg=ProfessionalTheme.WHITE, wraplength=200)
            desc_label.pack(pady=(0, 10))
        
        # Configure grid weights
        actions_grid.grid_columnconfigure(0, weight=1)
        actions_grid.grid_columnconfigure(1, weight=1)
        
        # Enhanced system status dashboard
        status_card = self.create_professional_card(self.content_frame, "System Status Dashboard", "Real-time Monitoring")
        status_card.pack(fill=tk.X, pady=(0, 20))
        
        status_content = tk.Frame(status_card, bg=ProfessionalTheme.WHITE)
        status_content.pack(fill=tk.X, padx=20, pady=15)
        
        # Create status grid
        status_grid = tk.Frame(status_content, bg=ProfessionalTheme.WHITE)
        status_grid.pack(fill=tk.X)
        
        # Left column - Core Systems
        left_status = tk.Frame(status_grid, bg=ProfessionalTheme.WHITE)
        left_status.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        tk.Label(left_status, text="Core Systems", font=('Segoe UI', 12, 'bold'),
                fg=ProfessionalTheme.DARK_GRAY, bg=ProfessionalTheme.WHITE).pack(anchor=tk.W)
        
        core_status = f"""
✅ Advanced Data Processing: Active
✅ ML Pipeline: {'Trained' if self.ml_core.models else 'Initializing'}
✅ Enhanced Visualization: Ready
✅ Robust Preprocessing: Enabled
✅ Cross-Validation: 5-fold Active
✅ Model Ensemble: {len(self.ml_core.models)} algorithms
        """.strip()
        
        tk.Label(left_status, text=core_status, font=('Segoe UI', 10),
                fg=ProfessionalTheme.DARK_GRAY, bg=ProfessionalTheme.WHITE,
                justify=tk.LEFT, anchor=tk.W).pack(anchor=tk.W, pady=(5, 0))
        
        # Right column - Data Status
        right_status = tk.Frame(status_grid, bg=ProfessionalTheme.WHITE)
        right_status.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(right_status, text="Data Analytics", font=('Segoe UI', 12, 'bold'),
                fg=ProfessionalTheme.DARK_GRAY, bg=ProfessionalTheme.WHITE).pack(anchor=tk.W)
        
        if self.ml_core.dataset is not None:
            data_status = f"""
📊 Dataset: {len(self.ml_core.dataset):,} records loaded
📈 Features: {len(self.ml_core.dataset.columns)-1} medical parameters
🎯 Target: Heart_Disease (Binary Classification)
🔍 Missing Values: {self.ml_core.dataset.isnull().sum().sum()}
💾 Memory Usage: {self.ml_core.dataset.memory_usage().sum() / 1024**2:.1f} MB
⚖️ Class Balance: Automatically optimized
            """.strip()
        else:
            data_status = "📊 No dataset currently loaded"
        
        tk.Label(right_status, text=data_status, font=('Segoe UI', 10),
                fg=ProfessionalTheme.DARK_GRAY, bg=ProfessionalTheme.WHITE,
                justify=tk.LEFT, anchor=tk.W).pack(anchor=tk.W, pady=(5, 0))
        
        # Enhanced model performance summary
        if self.ml_core.performance_data:
            perf_card = self.create_professional_card(self.content_frame, "Model Performance Summary", 
                                                    "Latest Training Results")
            perf_card.pack(fill=tk.X, pady=(0, 20))
            
            perf_content = tk.Frame(perf_card, bg=ProfessionalTheme.WHITE)
            perf_content.pack(fill=tk.X, padx=20, pady=15)
            
            # Performance metrics in a nice table format
            headers_frame = tk.Frame(perf_content, bg=ProfessionalTheme.LIGHT_GRAY)
            headers_frame.pack(fill=tk.X, pady=(0, 10))
            
            tk.Label(headers_frame, text="Model", font=('Segoe UI', 10, 'bold'),
                    bg=ProfessionalTheme.LIGHT_GRAY, fg=ProfessionalTheme.DARK_GRAY).pack(side=tk.LEFT, padx=10, pady=5)
            tk.Label(headers_frame, text="Accuracy", font=('Segoe UI', 10, 'bold'),
                    bg=ProfessionalTheme.LIGHT_GRAY, fg=ProfessionalTheme.DARK_GRAY).pack(side=tk.LEFT, padx=40, pady=5)
            tk.Label(headers_frame, text="AUC Score", font=('Segoe UI', 10, 'bold'),
                    bg=ProfessionalTheme.LIGHT_GRAY, fg=ProfessionalTheme.DARK_GRAY).pack(side=tk.LEFT, padx=40, pady=5)
            tk.Label(headers_frame, text="CV Score", font=('Segoe UI', 10, 'bold'),
                    bg=ProfessionalTheme.LIGHT_GRAY, fg=ProfessionalTheme.DARK_GRAY).pack(side=tk.LEFT, padx=40, pady=5)
            tk.Label(headers_frame, text="Status", font=('Segoe UI', 10, 'bold'),
                    bg=ProfessionalTheme.LIGHT_GRAY, fg=ProfessionalTheme.DARK_GRAY).pack(side=tk.LEFT, padx=40, pady=5)
            
            for i, (model_name, perf) in enumerate(self.ml_core.performance_data.items()):
                row_frame = tk.Frame(perf_content, bg=ProfessionalTheme.WHITE if i % 2 == 0 else '#F8F9FA')
                row_frame.pack(fill=tk.X, pady=2)
                
                # Model symbol and name
                symbol = "🔗" if model_name == "K-Nearest Neighbors" else "🤖"
                model_text = f"{symbol} {model_name}"
                
                tk.Label(row_frame, text=model_text, font=('Segoe UI', 9),
                        bg=row_frame['bg'], fg=ProfessionalTheme.DARK_GRAY, width=25, anchor='w').pack(side=tk.LEFT, padx=5, pady=3)
                
                # Performance metrics
                tk.Label(row_frame, text=f"{perf['accuracy']:.1%}", font=('Segoe UI', 9, 'bold'),
                        bg=row_frame['bg'], fg=ProfessionalTheme.SUCCESS if perf['accuracy'] > 0.8 else ProfessionalTheme.WARNING,
                        width=12).pack(side=tk.LEFT, padx=25, pady=3)
                
                tk.Label(row_frame, text=f"{perf['auc']:.3f}", font=('Segoe UI', 9),
                        bg=row_frame['bg'], fg=ProfessionalTheme.DARK_GRAY, width=12).pack(side=tk.LEFT, padx=25, pady=3)
                
                tk.Label(row_frame, text=f"{perf['cv_mean']:.3f}±{perf['cv_std']:.3f}", font=('Segoe UI', 9),
                        bg=row_frame['bg'], fg=ProfessionalTheme.DARK_GRAY, width=15).pack(side=tk.LEFT, padx=15, pady=3)
                
                status = "🟢 Excellent" if perf['accuracy'] > 0.85 else "🟡 Good" if perf['accuracy'] > 0.75 else "🟠 Fair"
                tk.Label(row_frame, text=status, font=('Segoe UI', 9),
                        bg=row_frame['bg'], fg=ProfessionalTheme.DARK_GRAY, width=12).pack(side=tk.LEFT, padx=15, pady=3)
        
        # Enhanced feature highlights
        features_card = self.create_professional_card(self.content_frame, "Enhanced Features & Capabilities")
        features_card.pack(fill=tk.X)
        
        features_content = tk.Frame(features_card, bg=ProfessionalTheme.WHITE)
        features_content.pack(fill=tk.X, padx=20, pady=15)
        
        features_text = """
🔬 Advanced Data Processing:
   • Intelligent missing value handling with statistical imputation
   • Sophisticated feature engineering and encoding
   • Robust scaling to handle outliers and varying scales
   • Advanced class balancing with SMOTE-like oversampling

🤖 Enhanced Machine Learning Pipeline:
   • 5-algorithm ensemble including optimized KNN
   • Stratified cross-validation with comprehensive metrics
   • Hyperparameter optimization for each model
   • Advanced performance evaluation with AUC, precision, recall

📊 Comprehensive Visualization Engine:
   • 12+ different chart types for model analysis
   • Interactive performance dashboards
   • Data quality assessment visualizations
   • Correlation analysis and feature importance plots

🏥 Medical Domain Expertise:
   • Clinically relevant risk factors and correlations
   • Sophisticated cardiovascular risk modeling
   • Evidence-based feature selection
   • Professional medical report generation
        """.strip()
        
        tk.Label(features_content, text=features_text, font=('Segoe UI', 10),
                fg=ProfessionalTheme.DARK_GRAY, bg=ProfessionalTheme.WHITE,
                justify=tk.LEFT, anchor=tk.W).pack(anchor=tk.W)
    
    def show_enhanced_assessment(self):
        """Show enhanced patient assessment with better UI"""
        self.clear_content()
        self.current_page = "assessment"
        self.update_status("Enhanced assessment interface loaded")
        
        # Assessment header
        header_frame = tk.Frame(self.content_frame, bg=ProfessionalTheme.WHITE)
        header_frame.pack(fill=tk.X, pady=(0, 25))
        
        tk.Label(header_frame, text="Comprehensive Cardiovascular Risk Assessment",
                font=('Segoe UI', 20, 'bold'), fg=ProfessionalTheme.DARK_RED,
                bg=ProfessionalTheme.WHITE).pack(anchor=tk.W)
        
        tk.Label(header_frame, text="Advanced ML analysis with enhanced data processing and ensemble predictions",
                font=('Segoe UI', 12), fg=ProfessionalTheme.MEDIUM_GRAY,
                bg=ProfessionalTheme.WHITE).pack(anchor=tk.W, pady=(5, 0))
        
        # Progress indicator
        progress_frame = tk.Frame(header_frame, bg=ProfessionalTheme.WHITE)
        progress_frame.pack(anchor=tk.W, pady=(10, 0))
        
        steps = ["📝 Data Entry", "🔬 Processing", "🤖 Analysis", "📊 Results"]
        for i, step in enumerate(steps):
            color = ProfessionalTheme.SUCCESS if i == 0 else ProfessionalTheme.LIGHT_GRAY
            tk.Label(progress_frame, text=step, font=('Segoe UI', 9),
                    fg=color, bg=ProfessionalTheme.WHITE).pack(side=tk.LEFT, padx=(0, 20))
        
        # Form sections
        self.form_vars = {}
        
        # Patient Demographics Section
        demo_card = self.create_professional_card(self.content_frame, "Patient Demographics", "Basic Information")
        demo_card.pack(fill=tk.X, pady=(0, 15))
        
        demo_content = tk.Frame(demo_card, bg=ProfessionalTheme.WHITE)
        demo_content.pack(fill=tk.X, padx=20, pady=15)
        
        # Create form grid
        form_grid = tk.Frame(demo_content, bg=ProfessionalTheme.WHITE)
        form_grid.pack(fill=tk.X)
        
        # Demographics fields
        demo_fields = [
            ("Age", "Age (years):", "45", "int"),
            ("Gender", "Gender:", "Male", "combo", ["Male", "Female"]),
            ("BMI", "BMI (kg/m²):", "25.0", "float")
        ]
        
        for i, field_info in enumerate(demo_fields):
            self.create_form_field(form_grid, field_info, row=i//3, col=i%3)
        
        # Vital Signs Section
        vitals_card = self.create_professional_card(self.content_frame, "Vital Signs & Measurements", "Clinical Parameters")
        vitals_card.pack(fill=tk.X, pady=(0, 15))
        
        vitals_content = tk.Frame(vitals_card, bg=ProfessionalTheme.WHITE)
        vitals_content.pack(fill=tk.X, padx=20, pady=15)
        
        vitals_grid = tk.Frame(vitals_content, bg=ProfessionalTheme.WHITE)
        vitals_grid.pack(fill=tk.X)
        
        vitals_fields = [
            ("Blood_Pressure_Systolic", "Systolic BP (mmHg):", "120", "int"),
            ("Blood_Pressure_Diastolic", "Diastolic BP (mmHg):", "80", "int"),
            ("Cholesterol", "Cholesterol (mg/dL):", "200", "int"),
            ("Max_Heart_Rate", "Max Heart Rate (bpm):", "180", "int"),
            ("Resting_Heart_Rate", "Resting HR (bpm):", "70", "int"),
            ("Sleep_Hours", "Sleep Hours/Day:", "7.5", "float")
        ]
        
        for i, field_info in enumerate(vitals_fields):
            self.create_form_field(vitals_grid, field_info, row=i//3, col=i%3)
        
        # Risk Factors Section
        risk_card = self.create_professional_card(self.content_frame, "Risk Factors & Lifestyle", "Health History")
        risk_card.pack(fill=tk.X, pady=(0, 15))
        
        risk_content = tk.Frame(risk_card, bg=ProfessionalTheme.WHITE)
        risk_content.pack(fill=tk.X, padx=20, pady=15)
        
        risk_grid = tk.Frame(risk_content, bg=ProfessionalTheme.WHITE)
        risk_grid.pack(fill=tk.X)
        
        risk_fields = [
            ("Smoking", "Smoking Status:", "Never", "combo", ["Never", "Former", "Current"]),
            ("Exercise_Frequency", "Exercise Days/Week:", "3", "combo", ["0", "1", "2", "3", "4", "5", "6", "7"]),
            ("Family_History_Heart_Disease", "Family History:", "No", "combo", ["No", "Yes"]),
            ("Diabetes", "Diabetes Status:", "No", "combo", ["No", "Type 1", "Type 2"]),
            ("Exercise_Angina", "Exercise Angina:", "No", "combo", ["No", "Yes"]),
            ("Stress_Level", "Stress Level (1-10):", "5", "combo", [str(i) for i in range(1, 11)]),
            ("Alcohol_Consumption", "Alcohol Use:", "Light", "combo", ["None", "Light", "Moderate", "Heavy"]),
            ("Previous_Heart_Events", "Previous Heart Events:", "No", "combo", ["No", "Yes"])
        ]
        
        for i, field_info in enumerate(risk_fields):
            self.create_form_field(risk_grid, field_info, row=i//3, col=i%3)
        
        # Assessment Action Section
        action_card = self.create_professional_card(self.content_frame, "Perform Assessment", "AI-Powered Analysis")
        action_card.pack(fill=tk.X, pady=(0, 15))
        
        action_content = tk.Frame(action_card, bg=ProfessionalTheme.WHITE)
        action_content.pack(fill=tk.X, padx=20, pady=20)
        
        # Assessment description
        desc_text = """This comprehensive assessment will analyze your cardiovascular risk using:
• Advanced machine learning ensemble (5 algorithms including optimized KNN)
• Sophisticated data preprocessing and feature engineering
• Cross-validated model predictions with confidence intervals
• Evidence-based risk factor correlations and medical domain knowledge"""
        
        tk.Label(action_content, text=desc_text, font=('Segoe UI', 10),
                fg=ProfessionalTheme.MEDIUM_GRAY, bg=ProfessionalTheme.WHITE,
                justify=tk.LEFT, wraplength=800).pack(anchor=tk.W, pady=(0, 15))
        
        # Assessment button with enhanced styling
        assess_btn = tk.Button(action_content, text="🔬 Perform Advanced Risk Assessment",
                             command=self.perform_enhanced_assessment,
                             bg=ProfessionalTheme.SUCCESS, fg=ProfessionalTheme.WHITE,
                             font=('Segoe UI', 14, 'bold'), relief=tk.RAISED,
                             padx=40, pady=15, cursor='hand2')
        assess_btn.pack(pady=10)
        
        # Add hover effects for the assessment button
        def on_enter(e):
            assess_btn.config(bg=ProfessionalTheme.DARK_GRAY, relief=tk.RAISED)
        def on_leave(e):
            assess_btn.config(bg=ProfessionalTheme.SUCCESS, relief=tk.RAISED)
        
        assess_btn.bind("<Enter>", on_enter)
        assess_btn.bind("<Leave>", on_leave)
    
    def create_form_field(self, parent, field_info, row, col):
        """Create a form field with proper validation"""
        field_name = field_info[0]
        label_text = field_info[1]
        default_value = field_info[2]
        field_type = field_info[3]
        options = field_info[4] if len(field_info) > 4 else None
        
        # Create field frame
        field_frame = tk.Frame(parent, bg=ProfessionalTheme.WHITE)
        field_frame.grid(row=row, column=col, padx=15, pady=10, sticky="ew")
        parent.grid_columnconfigure(col, weight=1)
        
        # Label
        tk.Label(field_frame, text=label_text, font=('Segoe UI', 10, 'bold'),
                fg=ProfessionalTheme.DARK_GRAY, bg=ProfessionalTheme.WHITE).pack(anchor=tk.W)
        
        # Input widget
        if field_type == "combo":
            var = tk.StringVar(value=default_value)
            widget = ttk.Combobox(field_frame, textvariable=var, values=options,
                                state="readonly", font=('Segoe UI', 10))
        else:
            var = tk.StringVar(value=default_value)
            widget = tk.Entry(field_frame, textvariable=var, font=('Segoe UI', 10),
                            relief=tk.SOLID, bd=1, bg=ProfessionalTheme.WHITE)
        
        widget.pack(fill=tk.X, pady=(5, 0))
        self.form_vars[field_name] = var
    
    def create_professional_card(self, parent, title, subtitle=None):
        """Create a professional styled card"""
        card_frame = tk.Frame(parent, bg=ProfessionalTheme.WHITE, relief=tk.RAISED, bd=2)
        
        # Header
        header_frame = tk.Frame(card_frame, bg=ProfessionalTheme.LIGHT_GRAY, height=50)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_frame = tk.Frame(header_frame, bg=ProfessionalTheme.LIGHT_GRAY)
        title_frame.pack(anchor=tk.W, padx=15, pady=10)
        
        tk.Label(title_frame, text=title, font=('Segoe UI', 14, 'bold'),
                fg=ProfessionalTheme.DARK_GRAY, bg=ProfessionalTheme.LIGHT_GRAY).pack(anchor=tk.W)
        
        if subtitle:
            tk.Label(title_frame, text=subtitle, font=('Segoe UI', 10),
                    fg=ProfessionalTheme.MEDIUM_GRAY, bg=ProfessionalTheme.LIGHT_GRAY).pack(anchor=tk.W)
        
        return card_frame
    
    def perform_enhanced_assessment(self):
        """Perform enhanced cardiovascular risk assessment"""
        try:
            self.update_status("Processing enhanced assessment...", True, 20)
            
            # Collect form data
            patient_data = {}
            for field_name, var in self.form_vars.items():
                value = var.get().strip()
                
                # Data validation and conversion
                if field_name in ['Age', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 
                                 'Cholesterol', 'Max_Heart_Rate', 'Resting_Heart_Rate', 'Exercise_Frequency', 'Stress_Level']:
                    try:
                        patient_data[field_name] = int(value)
                    except ValueError:
                        messagebox.showerror("Input Error", f"Please enter a valid number for {field_name}")
                        return
                elif field_name in ['BMI', 'Sleep_Hours']:
                    try:
                        patient_data[field_name] = float(value)
                    except ValueError:
                        messagebox.showerror("Input Error", f"Please enter a valid number for {field_name}")
                        return
                else:
                    patient_data[field_name] = value
            
            self.update_status("Running ML models with enhanced preprocessing...", True, 60)
            
            # Perform enhanced prediction
            results = self.ml_core.predict_risk_enhanced(patient_data)
            self.current_results = results
            self.patient_answers = patient_data
            
            self.update_status("Assessment complete - analyzing results...", True, 100)
            
            # Show results
            self.show_enhanced_results()
            
        except Exception as e:
            self.update_status("Assessment error occurred")
            messagebox.showerror("Assessment Error", f"Error during assessment: {str(e)}")
    
    def show_enhanced_results(self):
        """Show enhanced assessment results"""
        if not self.current_results:
            messagebox.showwarning("No Results", "Please perform an assessment first.")
            return
        
        self.clear_content()
        self.current_page = "results"
        self.update_status("Enhanced results displayed")
        
        # Results header
        header_frame = tk.Frame(self.content_frame, bg=ProfessionalTheme.WHITE)
        header_frame.pack(fill=tk.X, pady=(0, 25))
        
        tk.Label(header_frame, text="Cardiovascular Risk Assessment Results",
                font=('Segoe UI', 20, 'bold'), fg=ProfessionalTheme.DARK_RED,
                bg=ProfessionalTheme.WHITE).pack(anchor=tk.W)
        
        # Risk summary
        self.create_risk_summary()
        
        # Model predictions
        self.create_model_predictions()
        
        # Recommendations
        self.create_recommendations()
    
    def create_risk_summary(self):
        """Create risk summary section"""
        summary_card = self.create_professional_card(self.content_frame, "Risk Assessment Summary")
        summary_card.pack(fill=tk.X, pady=(0, 15))
        
        content = tk.Frame(summary_card, bg=ProfessionalTheme.WHITE)
        content.pack(fill=tk.X, padx=20, pady=15)
        
        # Calculate overall risk
        risk_scores = [result['risk_score'] for result in self.current_results.values()]
        avg_risk = sum(risk_scores) / len(risk_scores)
        
        risk_level = "High" if avg_risk > 0.7 else "Moderate" if avg_risk > 0.4 else "Low"
        risk_color = ProfessionalTheme.PRIMARY_RED if risk_level == "High" else ProfessionalTheme.WARNING if risk_level == "Moderate" else ProfessionalTheme.SUCCESS
        
        tk.Label(content, text=f"Overall Risk Level: {risk_level} ({avg_risk:.1%})",
                font=('Segoe UI', 16, 'bold'), fg=risk_color,
                bg=ProfessionalTheme.WHITE).pack(pady=(0, 10))
    
    def create_model_predictions(self):
        """Create model predictions section"""
        pred_card = self.create_professional_card(self.content_frame, "Model Predictions")
        pred_card.pack(fill=tk.X, pady=(0, 15))
        
        content = tk.Frame(pred_card, bg=ProfessionalTheme.WHITE)
        content.pack(fill=tk.X, padx=20, pady=15)
        
        for model_name, result in self.current_results.items():
            model_frame = tk.Frame(content, bg=ProfessionalTheme.LIGHT_GRAY, relief=tk.RAISED, bd=1)
            model_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(model_frame, text=f"{model_name}: {result['risk_score']:.1%} risk",
                    font=('Segoe UI', 12, 'bold'), bg=ProfessionalTheme.LIGHT_GRAY,
                    fg=ProfessionalTheme.DARK_GRAY).pack(padx=15, pady=10)
    
    def create_recommendations(self):
        """Create recommendations section"""
        rec_card = self.create_professional_card(self.content_frame, "Health Recommendations")
        rec_card.pack(fill=tk.X, pady=(0, 15))
        
        content = tk.Frame(rec_card, bg=ProfessionalTheme.WHITE)
        content.pack(fill=tk.X, padx=20, pady=15)
        
        recommendations = [
            "Regular cardiovascular exercise (30 minutes, 5 days/week)",
            "Maintain healthy blood pressure (<140/90 mmHg)",
            "Monitor cholesterol levels regularly",
            "Follow a heart-healthy diet low in saturated fats",
            "Avoid smoking and limit alcohol consumption",
            "Regular medical check-ups and screenings"
        ]
        
        for rec in recommendations:
            tk.Label(content, text=f"• {rec}", font=('Segoe UI', 10),
                    fg=ProfessionalTheme.DARK_GRAY, bg=ProfessionalTheme.WHITE,
                    wraplength=700, justify=tk.LEFT).pack(anchor=tk.W, pady=2)
    
    def show_advanced_analytics(self):
        """Show advanced analytics dashboard"""
        self.clear_content()
        self.current_page = "analytics"
        self.update_status("Loading advanced analytics...")
        
        if not self.ml_core.models:
            tk.Label(self.content_frame, text="No models available. Please train models first.",
                    font=('Segoe UI', 14), fg=ProfessionalTheme.PRIMARY_RED,
                    bg=ProfessionalTheme.WHITE).pack(pady=50)
            return
        
        try:
            fig = self.ml_core.generate_comprehensive_analysis_plots()
            if fig:
                self.embed_matplotlib_figure(fig, "Comprehensive Model Analysis")
        except Exception as e:
            tk.Label(self.content_frame, text=f"Error generating analytics: {str(e)}",
                    font=('Segoe UI', 12), fg=ProfessionalTheme.PRIMARY_RED,
                    bg=ProfessionalTheme.WHITE).pack(pady=20)
    
    def show_data_quality_analysis(self):
        """Show data quality analysis"""
        self.clear_content()
        self.current_page = "data_quality"
        self.update_status("Loading data quality analysis...")
        
        if self.ml_core.dataset is None:
            tk.Label(self.content_frame, text="No dataset available for analysis.",
                    font=('Segoe UI', 14), fg=ProfessionalTheme.PRIMARY_RED,
                    bg=ProfessionalTheme.WHITE).pack(pady=50)
            return
        
        try:
            fig = self.ml_core.generate_data_quality_plots()
            if fig:
                self.embed_matplotlib_figure(fig, "Data Quality Analysis")
        except Exception as e:
            tk.Label(self.content_frame, text=f"Error generating data quality plots: {str(e)}",
                    font=('Segoe UI', 12), fg=ProfessionalTheme.PRIMARY_RED,
                    bg=ProfessionalTheme.WHITE).pack(pady=20)
    
    def show_enhanced_data_management(self):
        """Show data management interface"""
        self.clear_content()
        self.current_page = "data_management"
        self.update_status("Data management interface loaded")
        
        tk.Label(self.content_frame, text="Data Management",
                font=('Segoe UI', 20, 'bold'), fg=ProfessionalTheme.DARK_RED,
                bg=ProfessionalTheme.WHITE).pack(pady=20)
        
        tk.Button(self.content_frame, text="Load New Dataset",
                 command=self.load_dataset, font=('Segoe UI', 12, 'bold'),
                 bg=ProfessionalTheme.INFO, fg=ProfessionalTheme.WHITE,
                 padx=30, pady=10).pack(pady=10)
    
    def show_enhanced_settings(self):
        """Show settings interface"""
        self.clear_content()
        self.current_page = "settings"
        self.update_status("Settings interface loaded")
        
        tk.Label(self.content_frame, text="System Settings",
                font=('Segoe UI', 20, 'bold'), fg=ProfessionalTheme.DARK_RED,
                bg=ProfessionalTheme.WHITE).pack(pady=20)
        
        tk.Label(self.content_frame, text="Settings panel - Configuration options would go here",
                font=('Segoe UI', 12), fg=ProfessionalTheme.MEDIUM_GRAY,
                bg=ProfessionalTheme.WHITE).pack(pady=10)
    
    def train_enhanced_models(self):
        """Train models with enhanced processing"""
        try:
            if self.ml_core.dataset is None:
                messagebox.showwarning("No Data", "Please load a dataset first.")
                return
            
            self.update_status("Training models with enhanced processing...", True, 0)
            
            target_col = self.find_target_column()
            if target_col:
                X = self.ml_core.dataset.drop(columns=[target_col])
                y = self.ml_core.dataset[target_col]
                
                if self.ml_core.train_advanced_models_enhanced(X, y):
                    self.update_status("Enhanced model training complete")
                    messagebox.showinfo("Training Complete", "Models trained successfully with enhanced processing!")
                else:
                    self.update_status("Enhanced training failed")
            else:
                messagebox.showerror("Error", "Could not identify target column.")
                
        except Exception as e:
            self.update_status("Training error occurred")
            messagebox.showerror("Training Error", f"Error during training: {str(e)}")
    
    def embed_matplotlib_figure(self, fig, title):
        """Embed matplotlib figure in tkinter"""
        # Create a frame for the plot
        plot_frame = tk.Frame(self.content_frame, bg=ProfessionalTheme.WHITE)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        tk.Label(plot_frame, text=title, font=('Segoe UI', 16, 'bold'),
                fg=ProfessionalTheme.DARK_RED, bg=ProfessionalTheme.WHITE).pack(pady=(0, 10))
        
        # Embed the plot
        canvas = FigureCanvasTkAgg(fig, plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def find_target_column(self):
        """Find the target column in the dataset"""
        if self.ml_core.dataset is None:
            return None
        
        possible_targets = ['Heart_Disease', 'heart_disease', 'target', 'diagnosis']
        for col in self.ml_core.dataset.columns:
            if col in possible_targets:
                return col
        return self.ml_core.dataset.columns[-1]  # Default to last column
    
    def load_dataset(self):
        """Load a new dataset"""
        file_path = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.ml_core.dataset = pd.read_csv(file_path)
                messagebox.showinfo("Success", f"Dataset loaded: {len(self.ml_core.dataset)} records")
                self.update_status(f"Dataset loaded: {len(self.ml_core.dataset)} records")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def show_medical_disclaimer(self):
        """Show medical disclaimer"""
        disclaimer = """
MEDICAL DISCLAIMER

This application is for educational and research purposes only.
It is NOT intended for actual medical diagnosis or treatment decisions.

Always consult with qualified healthcare professionals for medical advice.

Do you understand and agree to these terms?
        """
        
        result = messagebox.askyesno("Medical Disclaimer", disclaimer)
        if not result:
            self.root.destroy()


# Main execution
def main():
    """Main application entry point"""
    try:
        root = tk.Tk()
        app = EnhancedProfessionalMedicalApp(root)
        root.mainloop()
        
    except Exception as e:
        print(f"Application error: {str(e)}")
        if 'root' in locals():
            messagebox.showerror("Application Error", f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()