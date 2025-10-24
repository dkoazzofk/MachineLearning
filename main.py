import sys
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import variation
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QComboBox, QLabel, QFileDialog, QMessageBox, QTableWidget,
    QTableWidgetItem, QGroupBox, QProgressBar, QSizePolicy, QTextEdit,
    QSplitter, QFrame, QHeaderView, QSpinBox, QDoubleSpinBox, QFormLayout)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

matplotlib.use('Qt5Agg')

# ==================== CONFIGURATION ====================
@dataclass
class AnalysisConfig:
    """Configuration container for analysis parameters."""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    variance_threshold: float = 0.95
    alpha: float = 1.0
    pca_variance_threshold: float = 0.95
    target_column: str = "Y house price of unit area"

# ==================== DATA CLASSES ====================
@dataclass
class ModelResult:
    """Container for model results."""
    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    model: Any = None
    feature_importance: Optional[pd.Series] = None

@dataclass
class AnalysisData:
    """Container for analysis results."""
    data: pd.DataFrame
    target: str
    model_results: Dict[str, ModelResult] = field(default_factory=dict)
    pca_results: Dict[str, ModelResult] = field(default_factory=dict)
    pca_info: Dict[str, Any] = field(default_factory=dict)
    preprocessing_info: Dict[str, Any] = field(default_factory=dict)
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    vif_data: pd.DataFrame = field(default_factory=pd.DataFrame)

# ==================== LOGGING MIXIN ====================
class LoggingMixin:
    """Mixin class for adding logging capabilities."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def log_info(self, message: str):
        self.logger.info(message)

    def log_error(self, message: str):
        self.logger.error(f"{message}\n{traceback.format_exc()}")

# ==================== METRICS CALCULATION ====================
class MetricCalculator:
    """Optimized metrics calculator using vectorized operations."""
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAPE with clipping to avoid division by zero."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        epsilon = np.finfo(np.float64).eps
        y_true = np.where(y_true == 0, epsilon, y_true)
        return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Vectorized metric calculation for better performance."""
        try:
            y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
            epsilon = np.finfo(np.float64).eps
            
            mse = np.mean((y_true - y_pred) ** 2)
            mae = np.mean(np.abs(y_true - y_pred))
            
            safe_y_true = np.where(y_true == 0, epsilon, y_true)
            mape = np.mean(np.abs((safe_y_true - y_pred) / safe_y_true)) * 100
            
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + epsilon))
            
            return {
                'rmse': float(np.sqrt(mse)),
                'r2': float(r2),
                'mape': float(mape),
                'mae': float(mae)
            }
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            return {'rmse': 0.0, 'r2': 0.0, 'mape': 0.0, 'mae': 0.0}

# ==================== UNIFIED VISUALIZER ====================
class UnifiedVisualizer:
    """Unified visualization handler for all plot types."""
    def __init__(self):
        self.visualizers = {
            'correlation': self._plot_correlation,
            'distribution': self._plot_distribution,
            'pca': self._plot_pca
        }

    def visualize(self, plot_type: str, data: Any, canvas: FigureCanvas, **kwargs) -> None:
        """Generic visualization method."""
        if plot_type in self.visualizers:
            try:
                canvas.axes.clear()
                self.visualizers[plot_type](data, canvas, **kwargs)
                canvas.draw()
            except Exception as e:
                logging.error(f"Error in {plot_type} visualization: {str(e)}")
                canvas.axes.text(0.5, 0.5, 'Plotting Error', ha='center', va='center')
                canvas.draw()

    def _plot_correlation(self, data: pd.DataFrame, canvas: FigureCanvas) -> None:
        """Optimized correlation matrix plot."""
        mask = np.triu(np.ones_like(data, dtype=bool))
        sns.heatmap(data, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=canvas.axes, cbar_kws={'shrink': 0.8})
        canvas.axes.set_title('Feature Correlation Matrix')

    def _plot_distribution(self, data: pd.DataFrame, canvas: FigureCanvas, target: str) -> None:
        """Optimized distribution plot."""
        if target in data.columns:
            canvas.axes.hist(data[target], bins=15, alpha=0.7, 
                           edgecolor='black', color='skyblue', density=True)
            canvas.axes.set_title(f'Distribution of {target}')
            canvas.axes.set_xlabel(target)
            canvas.axes.set_ylabel('Density')

    def _plot_pca(self, pca_info: Dict[str, Any], canvas: FigureCanvas) -> None:
        """Optimized PCA variance plot."""
        explained_var = pca_info['explained_variance_ratio']
        cumulative_var = pca_info['cumulative_variance']
        x = range(1, len(explained_var) + 1)
        
        canvas.axes.bar(x, explained_var, alpha=0.6, color='blue', label='Individual')
        canvas.axes.step(x, cumulative_var, where='mid', color='red', label='Cumulative')
        canvas.axes.set_xlabel('Principal Components')
        canvas.axes.set_ylabel('Explained Variance Ratio')
        canvas.axes.set_title('PCA Explained Variance')
        canvas.axes.legend()
        canvas.axes.grid(True, alpha=0.3)

# ==================== SMART PREPROCESSOR ====================
class SmartPreprocessor(LoggingMixin):
    """Optimized preprocessing with intelligent feature handling."""
    def __init__(self, config: AnalysisConfig):
        super().__init__()
        self.config = config
        self.preprocessor = None

    def preprocess(self, data: pd.DataFrame, target: str) -> Tuple[np.ndarray, pd.Series, pd.DataFrame]:
        """Optimized preprocessing pipeline."""
        self.log_info("Starting data preprocessing")
        try:
            # Clean and encode data
            data_clean = self._clean_data(data)
            data_encoded = self._encode_categorical(data_clean)
            
            X = data_encoded.drop(columns=[target])
            y = data_encoded[target]
            
            # Feature selection and VIF calculation
            numerical_features = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numerical_features]
            vif_data = self._calculate_vif(X_numeric)
            
            # Remove low-variance features
            selector = VarianceThreshold(threshold=0.01)
            X_filtered = selector.fit_transform(X_numeric)
            selected_features = numerical_features[selector.get_support()]
            
            # Create preprocessing pipeline
            self.preprocessor = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            X_processed = self.preprocessor.fit_transform(X_filtered)
            self.log_info(f"Processed data shape: {X_processed.shape}")
            
            return X_processed, y, vif_data
            
        except Exception as e:
            self.log_error(f"Data preprocessing failed: {str(e)}")
            raise

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized data cleaning."""
        return df.fillna(df.select_dtypes(include=[np.number]).median())

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized categorical encoding."""
        cat_cols = df.select_dtypes(include=['object']).columns
        if not cat_cols.empty:
            return pd.get_dummies(df, columns=cat_cols, drop_first=True)
        return df

    def _calculate_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        """Vectorized VIF calculation."""
        try:
            vif_series = pd.Series([
                variance_inflation_factor(X.values, i) 
                for i in range(X.shape[1])
            ], index=X.columns)
            
            return pd.DataFrame({'Feature': vif_series.index, 'VIF': vif_series.values})
        except Exception as e:
            self.log_error(f"VIF calculation failed: {str(e)}")
            return pd.DataFrame()

# ==================== MODEL FACTORY ====================
class ModelFactory:
    """Factory for creating and evaluating models."""
    MODEL_CONFIGS = {
        'Linear Regression': lambda config: LinearRegression(),
        'Ridge Regression': lambda config: Ridge(alpha=config.alpha)
    }

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.cv_strategy = KFold(n_splits=min(config.cv_folds, 5), shuffle=True, 
                                random_state=config.random_state)

    def create_and_evaluate_models(self, X_train: np.ndarray, X_test: np.ndarray,
                                 y_train: pd.Series, y_test: pd.Series, prefix: str) -> Dict[str, ModelResult]:
        """Factory method for model creation and evaluation."""
        results = {}
        for name, model_func in self.MODEL_CONFIGS.items():
            try:
                model = model_func(self.config)
                result = self._evaluate_single_model(model, X_train, X_test, y_train, y_test)
                results[f"{prefix} {name}"] = result
            except Exception as e:
                logging.error(f"Failed to evaluate {name}: {str(e)}")
                results[f"{prefix} {name}"] = ModelResult(
                    name=f"{prefix} {name}",
                    metrics={k: 0.0 for k in ['rmse', 'r2', 'mape', 'mae', 
                                            'rmse_test', 'r2_test', 'mape_test', 'mae_test']}
                )
        return results

    def _evaluate_single_model(self, model: Any, X_train: np.ndarray, X_test: np.ndarray,
                             y_train: pd.Series, y_test: pd.Series) -> ModelResult:
        """Evaluate a single model with cross-validation."""
        # Cross-validation
        cv_results = cross_validate(model, X_train, y_train, cv=self.cv_strategy,
                                  scoring=['neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_error'])
        
        # Train final model
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = MetricCalculator.calculate_all_metrics(y_train, y_pred_train)
        test_metrics = MetricCalculator.calculate_all_metrics(y_test, y_pred_test)
        
        # Combine metrics
        metrics = train_metrics
        metrics.update({f"{k}_test": v for k, v in test_metrics.items()})
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'coef_'):
            feature_importance = pd.Series(model.coef_, 
                                         index=[f'Feature_{i}' for i in range(len(model.coef_))])
        
        return ModelResult(
            name=model.__class__.__name__,
            metrics=metrics,
            cv_mean=float(np.mean(cv_results['test_r2'])),
            cv_std=float(np.std(cv_results['test_r2'])),
            model=model,
            feature_importance=feature_importance
        )

# ==================== PCA ANALYZER ====================
class PCAAnalyzer(LoggingMixin):
    """Enhanced PCA analysis with comprehensive reporting."""
    def __init__(self, variance_threshold: float = 0.95):
        super().__init__()
        self.variance_threshold = variance_threshold
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=variance_threshold))
        ])

    def analyze(self, X: np.ndarray) -> Dict[str, Any]:
        """Perform PCA analysis with detailed diagnostics."""
        self.log_info("Starting PCA analysis")
        try:
            X_pca = self.pipeline.fit_transform(X)
            pca = self.pipeline.named_steps['pca']

            return {
                'transformed_data': X_pca,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'components': pca.components_,
                'n_components': pca.n_components_,
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
                'pipeline': self.pipeline
            }
        except Exception as e:
            self.log_error(f"PCA analysis failed: {str(e)}")
            raise

# ==================== MAIN ANALYSIS ENGINE ====================
class RegressionAnalyzer(LoggingMixin):
    """Main regression analysis engine."""
    def __init__(self, config: Optional[AnalysisConfig] = None):
        super().__init__()
        self.config = config or AnalysisConfig()

    def analyze(self, data: pd.DataFrame, target: str) -> AnalysisData:
        """Complete analysis pipeline."""
        self.log_info("Starting complete analysis pipeline")
        try:
            # Data preprocessing
            preprocessor = SmartPreprocessor(self.config)
            X_processed, y, vif_data = preprocessor.preprocess(data, target)

            # Data analysis
            numerical_data = data.select_dtypes(include=[np.number])
            corr_matrix = numerical_data.corr()

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=self.config.test_size, 
                random_state=self.config.random_state
            )

            # Train models on original data
            model_factory = ModelFactory(self.config)
            model_results = model_factory.create_and_evaluate_models(
                X_train, X_test, y_train, y_test, "Original"
            )

            # PCA analysis
            pca_analyzer = PCAAnalyzer(self.config.pca_variance_threshold)
            pca_results = pca_analyzer.analyze(X_train)

            # Transform test data using PCA
            X_test_pca = pca_results['pipeline'].transform(X_test)

            # Train models on PCA components
            pca_model_results = model_factory.create_and_evaluate_models(
                pca_results['transformed_data'], X_test_pca, y_train, y_test, "PCA"
            )

            analysis_data = AnalysisData(
                data=data,
                target=target,
                model_results=model_results,
                pca_results=pca_model_results,
                pca_info=pca_results,
                preprocessing_info={'preprocessor': preprocessor},
                correlation_matrix=corr_matrix,
                vif_data=vif_data
            )

            self.log_info("Analysis completed successfully")
            return analysis_data

        except Exception as e:
            self.log_error(f"Analysis pipeline failed: {str(e)}")
            raise

# ==================== GUI COMPONENTS ====================
class PlotCanvas(FigureCanvas):
    """Enhanced matplotlib canvas."""
    def __init__(self, parent=None, width: int = 5, height: int = 4, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def clear_plot(self):
        """Clear the plot."""
        try:
            self.fig.clear()
            self.axes = self.fig.add_subplot(111)
            self.draw()
        except Exception as e:
            logging.error(f"Error clearing plot: {str(e)}")

class AnalysisThread(QThread):
    """Background thread for analysis."""
    progress_updated = Signal(int)
    analysis_finished = Signal(AnalysisData)
    error_occurred = Signal(str)
    log_message = Signal(str)

    def __init__(self, file_path: str, target_col: str, config: AnalysisConfig):
        super().__init__()
        self.file_path = file_path
        self.target_col = target_col
        self.config = config

    def run(self):
        """Execute analysis in background thread."""
        try:
            self.log_message.emit("Starting analysis...")
            self.progress_updated.emit(10)

            data = self._load_data()
            self.log_message.emit(f"Loaded dataset with {len(data)} rows")
            self.progress_updated.emit(30)

            if self.target_col not in data.columns:
                raise ValueError(f"Target column '{self.target_col}' not found")

            self.log_message.emit("Preprocessing data...")
            self.progress_updated.emit(50)

            analyzer = RegressionAnalyzer(self.config)
            result = analyzer.analyze(data, self.target_col)
            
            self.log_message.emit("Analysis completed successfully")
            self.progress_updated.emit(90)
            self.analysis_finished.emit(result)
            self.progress_updated.emit(100)

        except Exception as e:
            self.log_message.emit(f"Error: {str(e)}")
            self.error_occurred.emit(f"Analysis failed: {str(e)}")

    def _load_data(self) -> pd.DataFrame:
        """Load data from file."""
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            if path.suffix.lower() == '.csv':
                return pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

class SettingsWidget(QWidget):
    """Configuration settings widget."""
    def __init__(self, config: AnalysisConfig):
        super().__init__()
        self.config = config
        self._setup_ui()

    def _setup_ui(self):
        """Initialize settings UI."""
        layout = QFormLayout()

        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.1, 0.5)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setValue(self.config.test_size)
        layout.addRow("Test Size:", self.test_size_spin)

        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(2, 10)
        self.cv_folds_spin.setValue(self.config.cv_folds)
        layout.addRow("CV Folds:", self.cv_folds_spin)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.1, 10.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(self.config.alpha)
        layout.addRow("Ridge Alpha:", self.alpha_spin)

        self.pca_threshold_spin = QDoubleSpinBox()
        self.pca_threshold_spin.setRange(0.7, 1.0)
        self.pca_threshold_spin.setSingleStep(0.05)
        self.pca_threshold_spin.setValue(self.config.pca_variance_threshold)
        layout.addRow("PCA Variance Threshold:", self.pca_threshold_spin)

        self.setLayout(layout)

    def get_config(self) -> AnalysisConfig:
        """Get updated configuration."""
        return AnalysisConfig(
            test_size=self.test_size_spin.value(),
            cv_folds=self.cv_folds_spin.value(),
            alpha=self.alpha_spin.value(),
            pca_variance_threshold=self.pca_threshold_spin.value(),
            target_column=self.config.target_column
        )

class DataAnalysisTab(QWidget):
    """Data analysis and visualization tab."""
    def __init__(self, controller: 'Controller'):
        super().__init__()
        self.controller = controller
        self.visualizer = UnifiedVisualizer()
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()

        # Control panel
        control_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Dataset")
        self.analyze_button = QPushButton("Analyze")
        self.target_combo = QComboBox()
        self.progress_bar = QProgressBar()

        self.load_button.clicked.connect(self.controller.load_data)
        self.analyze_button.clicked.connect(self.controller.analyze)

        control_layout.addWidget(self.load_button)
        control_layout.addWidget(QLabel("Target Variable:"))
        control_layout.addWidget(self.target_combo)
        control_layout.addWidget(self.analyze_button)
        control_layout.addStretch()

        layout.addLayout(control_layout)
        layout.addWidget(self.progress_bar)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setMaximumHeight(100)
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        # Visualization tabs
        self.plot_tabs = QTabWidget()

        # Correlation matrix tab
        self.corr_canvas = PlotCanvas(self, 8, 6)
        corr_widget = QWidget()
        corr_layout = QVBoxLayout()
        corr_layout.addWidget(self.corr_canvas)
        corr_widget.setLayout(corr_layout)
        self.plot_tabs.addTab(corr_widget, "Correlation Matrix")

        # Distributions tab
        self.dist_canvas = PlotCanvas(self, 8, 6)
        dist_widget = QWidget()
        dist_layout = QVBoxLayout()
        dist_layout.addWidget(self.dist_canvas)
        dist_widget.setLayout(dist_layout)
        self.plot_tabs.addTab(dist_widget, "Distributions")

        # PCA analysis tab
        self.pca_canvas = PlotCanvas(self, 8, 6)
        pca_widget = QWidget()
        pca_layout = QVBoxLayout()
        pca_layout.addWidget(self.pca_canvas)
        pca_widget.setLayout(pca_layout)
        self.plot_tabs.addTab(pca_widget, "PCA Analysis")

        # VIF tab
        self.vif_text = QTextEdit()
        self.vif_text.setReadOnly(True)
        vif_widget = QWidget()
        vif_layout = QVBoxLayout()
        vif_layout.addWidget(QLabel("VIF Analysis (Multicollinearity):"))
        vif_layout.addWidget(self.vif_text)
        vif_widget.setLayout(vif_layout)
        self.plot_tabs.addTab(vif_widget, "VIF Analysis")

        layout.addWidget(self.plot_tabs)
        self.setLayout(layout)

    def clear_all_plots(self):
        """Clear all plots completely before new analysis."""
        try:
            self.corr_canvas.clear_plot()
            self.dist_canvas.clear_plot()
            self.pca_canvas.clear_plot()
            self.vif_text.clear()
        except Exception as e:
            logging.error(f"Error clearing plots: {str(e)}")

    def update_data(self, analysis_data: AnalysisData):
        """Update plots with new analysis data."""
        try:
            self.clear_all_plots()

            # Update visualizations
            self.visualizer.visualize('correlation', analysis_data.correlation_matrix, self.corr_canvas)
            self.visualizer.visualize('distribution', analysis_data.data, self.dist_canvas, 
                                    target=analysis_data.target)
            self.visualizer.visualize('pca', analysis_data.pca_info, self.pca_canvas)

            # Display VIF analysis
            self._display_vif_analysis(analysis_data.vif_data)

        except Exception as e:
            logging.error(f"Error updating plots: {str(e)}")

    def _display_vif_analysis(self, vif_data: pd.DataFrame):
        """Display VIF analysis results."""
        try:
            vif_text = "Variance Inflation Factor Analysis\n"
            vif_text += "=" * 40 + "\n\n"
            vif_text += "VIF Interpretation:\n"
            vif_text += "VIF = 1: No correlation\n"
            vif_text += "1 < VIF < 5: Moderate correlation\n"
            vif_text += "VIF >= 5: High correlation\n"
            vif_text += "VIF >= 10: Severe multicollinearity\n\n"
            vif_text += vif_data.to_string()

            self.vif_text.setText(vif_text)
        except Exception as e:
            logging.error(f"Error displaying VIF analysis: {str(e)}")

    def log_message(self, message: str):
        """Add message to log output."""
        self.log_output.append(f"{message}")

class ResultsTab(QWidget):
    """Results display tab."""
    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()

        # Original models results
        self.original_group = QGroupBox("Standard Models (Original Features)")
        original_layout = QVBoxLayout()
        self.original_table = self._create_table()
        original_layout.addWidget(self.original_table)
        self.original_group.setLayout(original_layout)

        # PCA models results
        self.pca_group = QGroupBox("PCA Models (Principal Components)")
        pca_layout = QVBoxLayout()
        self.pca_table = self._create_table()
        pca_layout.addWidget(self.pca_table)
        self.pca_group.setLayout(pca_layout)

        layout.addWidget(self.original_group)
        layout.addWidget(self.pca_group)
        self.setLayout(layout)

    def _create_table(self) -> QTableWidget:
        """Create a standardized results table."""
        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels([
            "Model", "R² (Train)", "R² (Test)", "RMSE", "MAPE", "MAE", "CV Score"
        ])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return table

    def update_results(self, analysis_data: AnalysisData):
        """Update results tables with new analysis data."""
        self._populate_table(self.original_table, analysis_data.model_results)
        self._populate_table(self.pca_table, analysis_data.pca_results)

    def _populate_table(self, table: QTableWidget, results: Dict[str, ModelResult]):
        """Populate table with model results."""
        table.setRowCount(len(results))

        for row, (name, result) in enumerate(results.items()):
            table.setItem(row, 0, QTableWidgetItem(name))
            table.setItem(row, 1, QTableWidgetItem(f"{result.metrics.get('r2', 0):.4f}"))
            table.setItem(row, 2, QTableWidgetItem(f"{result.metrics.get('r2_test', 0):.4f}"))
            table.setItem(row, 3, QTableWidgetItem(f"{result.metrics.get('rmse_test', 0):.4f}"))
            table.setItem(row, 4, QTableWidgetItem(f"{result.metrics.get('mape_test', 0):.2f}%"))
            table.setItem(row, 5, QTableWidgetItem(f"{result.metrics.get('mae_test', 0):.4f}"))
            table.setItem(row, 6, QTableWidgetItem(f"{result.cv_mean:.4f} ± {result.cv_std:.4f}"))

# ==================== MAIN WINDOW ====================
class MainWindow(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.controller = Controller(self)
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the main window UI."""
        self.setWindowTitle("Regression Analyzer - Linear Regression Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Settings panel
        settings_frame = QFrame()
        settings_frame.setFrameStyle(QFrame.StyledPanel)
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Analysis Settings:"))
        self.settings_widget = SettingsWidget(self.controller.config)
        settings_layout.addWidget(self.settings_widget)
        settings_layout.addStretch()
        settings_frame.setLayout(settings_layout)
        layout.addWidget(settings_frame)

        self.tab_widget = QTabWidget()

        self.data_tab = DataAnalysisTab(self.controller)
        self.tab_widget.addTab(self.data_tab, "Data Analysis")

        self.results_tab = ResultsTab()
        self.tab_widget.addTab(self.results_tab, "Model Results")

        layout.addWidget(self.tab_widget)

    def update_target_variables(self, columns: List[str]):
        """Update target variable dropdown."""
        self.data_tab.target_combo.clear()
        self.data_tab.target_combo.addItems(columns)

    def update_progress(self, value: int):
        """Update progress bar."""
        self.data_tab.progress_bar.setValue(value)

    def log_message(self, message: str):
        """Add message to log output."""
        self.data_tab.log_message(message)

# ==================== CONTROLLER ====================
class Controller(LoggingMixin):
    """MVC Controller for coordinating model, view, and user interactions."""
    def __init__(self, view: MainWindow):
        super().__init__()
        self.view = view
        self.current_file_path = None
        self.analysis_thread = None
        self.config = AnalysisConfig()

    def load_data(self):
        """Handle data loading from file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self.view, "Open Dataset", "", "CSV Files (*.csv);;All Files (*)")

            if file_path:
                self.current_file_path = file_path
                data = self._load_data_file(file_path)

                numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()

                if not numerical_columns:
                    raise ValueError("No numerical columns found in the dataset")

                self.view.update_target_variables(numerical_columns)
                self.view.log_message(f"✓ Loaded dataset: {len(data)} rows, {len(data.columns)} columns")

        except Exception as e:
            self.log_error(f"Data loading failed: {str(e)}")
            QMessageBox.critical(self.view, "Error", f"Failed to load data: {str(e)}")

    def analyze(self):
        """Start analysis process."""
        if not self._validate_inputs():
            return

        self._prepare_ui_for_analysis()
        self._start_analysis_thread()

    def _validate_inputs(self) -> bool:
        """Validate all inputs efficiently."""
        if not self.current_file_path:
            QMessageBox.warning(self.view, "Warning", "Please load a dataset first")
            return False
        
        if not self.view.data_tab.target_combo.currentText():
            QMessageBox.warning(self.view, "Warning", "Please select a target variable")
            return False
        
        return True

    def _prepare_ui_for_analysis(self):
        """Prepare UI for analysis."""
        self.config = self.view.settings_widget.get_config()
        self.view.data_tab.clear_all_plots()
        self.view.data_tab.analyze_button.setEnabled(False)
        self.view.data_tab.log_output.clear()

    def _start_analysis_thread(self):
        """Start analysis thread."""
        self.analysis_thread = AnalysisThread(
            self.current_file_path,
            self.view.data_tab.target_combo.currentText(),
            self.config
        )
        
        # Connect signals
        self.analysis_thread.progress_updated.connect(self.view.update_progress)
        self.analysis_thread.analysis_finished.connect(self.on_analysis_complete)
        self.analysis_thread.error_occurred.connect(self.on_analysis_error)
        self.analysis_thread.log_message.connect(self.view.log_message)
        
        self.analysis_thread.start()

    def on_analysis_complete(self, results: AnalysisData):
        """Handle completed analysis."""
        try:
            self.view.data_tab.update_data(results)
            self.view.results_tab.update_results(results)
            self.view.data_tab.analyze_button.setEnabled(True)
            self.view.log_message("✓ Analysis completed successfully!")

            QMessageBox.information(
                self.view, "Analysis Complete",
                "Model training and evaluation completed successfully!\n"
                "Check the Results tab for detailed metrics."
            )

        except Exception as e:
            self.log_error(f"Error handling analysis results: {str(e)}")
            QMessageBox.critical(self.view, "Error", f"Error processing results: {str(e)}")
            self.view.data_tab.analyze_button.setEnabled(True)

    def on_analysis_error(self, error_message: str):
        """Handle analysis errors."""
        self.log_error(f"Analysis error: {error_message}")
        self.view.data_tab.analyze_button.setEnabled(True)
        self.view.log_message(f"✗ {error_message}")
        QMessageBox.critical(self.view, "Analysis Error", error_message)

    def _load_data_file(self, file_path: str) -> pd.DataFrame:
        """Load data from file with robust error handling."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if path.suffix.lower() == '.csv':
                return pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        except pd.errors.EmptyDataError:
            raise Exception("The file appears to be empty")
        except pd.errors.ParserError as e:
            raise Exception(f"Error parsing file: {str(e)}")
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

# ==================== APPLICATION ENTRY POINT ====================
def setup_logging():
    """Setup application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('regression_analyzer.log', encoding='utf-8'),
            logging.StreamHandler()
        ])

def main():
    """Main application entry point."""
    setup_logging()

    app = QApplication(sys.argv)
    app.setApplicationName("Regression Analyzer")
    app.setApplicationVersion("1.0.0")
    app.setStyle('Fusion')

    try:
        window = MainWindow()
        window.show()

        window.log_message("Welcome to Regression Analyzer v1.0!")
        window.log_message("1. Load a dataset (CSV)")
        window.log_message("2. Select target variable")
        window.log_message("3. Click Analyze to start")

        return app.exec()

    except Exception as e:
        logging.critical(f"Application failed to start: {str(e)}")
        QMessageBox.critical(None, "Fatal Error", f"Application failed to start:\n{str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())