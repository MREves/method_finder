"""Machine Learning & Predictive Analytics Decision Schema & Taxonomy.

Defines the complete methodological taxonomy, decision nodes, model architectures,
algorithmic fairness metrics, time series models, and clinical/operational scenarios.

Covers:
1. Supervised Classification (Penalized Logistic, CART, Random Forest, Gradient Boosting, SVM, k-NN)
2. Supervised Regression (Ridge, Lasso, ElasticNet, Random Forest Regressor, Gradient Boosting Regressor, SVR)
3. Time Series & Operational Demand Forecasting (Holt-Winters, SARIMA, Lagged Tree Ensembles)
4. Unsupervised Clustering & Dimensionality Reduction (K-Means, DBSCAN, Hierarchical, PCA)
5. Algorithmic Fairness & Equity Analysis (Demographic Parity, Disparate Impact, Equalized Odds)
"""

from typing import Dict, Any, List

# =====================================================================
# 1. High-Level Machine Learning Task Families
# =====================================================================

ML_TASK_FAMILIES: Dict[str, Dict[str, Any]] = {
    "classification": {
        "name": "Supervised Classification",
        "description": "Predicting categorical outcomes, clinical event occurrence, or operational status (e.g. readmission, DNA risk).",
        "icon": "🎯",
        "output_type": "Categorical (Binary or Multi-Class Probabilities)",
        "default_metric": "PR-AUC / F1-Score (for imbalanced classes) or ROC-AUC"
    },
    "regression": {
        "name": "Supervised Continuous Regression",
        "description": "Predicting numeric quantities, length of stay, clinical cost, or biomarker trajectory.",
        "icon": "📈",
        "output_type": "Continuous Numerical Value",
        "default_metric": "RMSE / MAE / R²"
    },
    "time_series": {
        "name": "Time Series & Operational Demand Forecasting",
        "description": "Predicting forward-looking hospital attendances, bed demand, or workforce sickness trends across temporal horizons.",
        "icon": "⏱️",
        "output_type": "Sequential Numerical Forecast with Prediction Intervals",
        "default_metric": "MAE / RMSE / MAPE (via Expanding Window / Rolling Origin)"
    },
    "clustering": {
        "name": "Unsupervised Clustering & Subtyping",
        "description": "Discovering latent patient subgroups, spatial disease clusters, or genomic outbreak isolates without predefined labels.",
        "icon": "🔍",
        "output_type": "Cluster Assignments & Density Profiles",
        "default_metric": "Silhouette Coefficient / Davies-Bouldin / Elbow WCSS"
    },
    "dim_reduction": {
        "name": "Dimensionality Reduction & Feature Extraction",
        "description": "Compressing high-dimensional clinical or genomic feature spaces into orthogonal summary axes for visualization and denoising.",
        "icon": "📉",
        "output_type": "Reduced Component Embeddings (e.g. PC1, PC2)",
        "default_metric": "Cumulative Explained Variance Ratio"
    },
    "fairness_audit": {
        "name": "Algorithmic Fairness & Health Equity Audit",
        "description": "Auditing operational and clinical algorithms for disparate impact and demographic bias across protected patient groups.",
        "icon": "⚖️",
        "output_type": "Disparity Ratios & Group Parity Differences",
        "default_metric": "Disparate Impact Ratio (80% Rule) & Equalized Odds Parity"
    }
}


# =====================================================================
# 2. Comprehensive Model Taxonomy & Parameter Catalog
# =====================================================================

ML_MODELS: Dict[str, Dict[str, Any]] = {
    # --- Classification Models ---
    "penalized_logistic_regression": {
        "id": "penalized_logistic_regression",
        "name": "Penalized Logistic Regression (L1 / L2 / ElasticNet)",
        "family": "classification",
        "short_desc": "Linear probability estimator regularized by L1 (Lasso) or L2 (Ridge) penalties to prevent overfitting and perform feature selection.",
        "interpretability": "High",
        "interpretability_score": 5,
        "sample_efficiency": "High (effective even when N < 500)",
        "handles_non_linear": False,
        "handles_imbalance": "Via Class Weights (`class_weight='balanced'`)",
        "primary_use_case": "Clinical risk prediction where odds ratios and model transparency are required by clinicians and regulatory bodies.",
        "loss_function": r"\mathcal{L}(\beta) = -\sum_{i=1}^n \left[ y_i \ln(p_i) + (1 - y_i)\ln(1 - p_i) \right] + \lambda \left( \alpha \|\beta\|_1 + \frac{1 - \alpha}{2} \|\beta\|_2^2 \right)",
        "loss_explanation": "Standard binary cross-entropy (log-loss) augmented with an ElasticNet penalty. L1 (Lasso) shrinks irrelevant coefficients strictly to zero for sparse feature selection, while L2 (Ridge) stabilizes collinear clinical predictors.",
        "hyperparameters": [
            {"param": "C", "name": "Inverse Regularization Strength", "default": "1.0", "range": "0.01 - 100.0", "description": "Smaller values specify stronger shrinkage to combat overfitting."},
            {"param": "penalty", "name": "Penalty Type", "default": "'elasticnet'", "options": "['l1', 'l2', 'elasticnet', 'none']", "description": "L1 for feature sparsity; L2 for grouping collinear features."},
            {"param": "l1_ratio", "name": "ElasticNet Mixing Ratio", "default": "0.5", "range": "0.0 - 1.0", "description": "0 corresponds to pure Ridge (L2); 1 corresponds to pure Lasso (L1)."}
        ],
        "pitfalls": [
            "Fails to capture non-linear thresholds or complex multi-way feature interactions without manual feature engineering.",
            "Requires continuous predictors to be standardized (z-scored) prior to fitting; otherwise, unscaled features are unfairly penalized."
        ],
        "pros": ["Direct interpretation via Odds Ratios", "Rapid training and inference", "Resistant to overfitting in small clinical cohorts"],
        "cons": ["Linear decision boundary only", "Sensitive to extreme outliers"]
    },

    "decision_tree_classifier": {
        "id": "decision_tree_classifier",
        "name": "CART Decision Tree Classifier",
        "family": "classification",
        "short_desc": "Single recursive binary partitioning tree splitting on orthogonal feature thresholds to maximize Gini purity or information gain.",
        "interpretability": "Very High",
        "interpretability_score": 5,
        "sample_efficiency": "Moderate",
        "handles_non_linear": True,
        "handles_imbalance": "Via Class Weights (`class_weight='balanced'`)",
        "primary_use_case": "Clinical triage rules and bedside decision protocols requiring simple 'if-then' threshold logic.",
        "loss_function": r"Gini(D) = 1 - \sum_{k=1}^K p_k^2 \quad \text{or} \quad H(D) = -\sum_{k=1}^K p_k \log_2(p_k)",
        "loss_explanation": "At each node, the split is selected to minimize Gini impurity or Shannon entropy, isolating homogeneous classes in each sub-branch.",
        "hyperparameters": [
            {"param": "max_depth", "name": "Maximum Tree Depth", "default": "4", "range": "2 - 10", "description": "Limits tree growth to maintain human readability and prevent catastrophic overfitting."},
            {"param": "min_samples_split", "name": "Min Samples to Split Node", "default": "20", "range": "5 - 50", "description": "Prevents splits on idiosyncratic small subsets of patients."}
        ],
        "pitfalls": [
            "High variance: minor fluctuations in training data can lead to completely different tree structures.",
            "Axis-aligned splits struggle with linear diagonal relationships."
        ],
        "pros": ["Visualizable bedside flowcharts", "Invariant to feature scaling", "Handles categorical & continuous features naturally"],
        "cons": ["Prone to severe overfitting without pruning", "Lower predictive power than ensemble models"]
    },

    "random_forest_classifier": {
        "id": "random_forest_classifier",
        "name": "Random Forest Classifier (Bagged Ensemble)",
        "family": "classification",
        "short_desc": "Ensemble of decorrelated decision trees built on bootstrap samples with random feature sub-sampling, aggregating predictions via majority vote.",
        "interpretability": "Moderate",
        "interpretability_score": 3,
        "sample_efficiency": "High (robust across medium and large datasets)",
        "handles_non_linear": True,
        "handles_imbalance": "Via Balanced Sub-sampling (`class_weight='balanced_subsample'`)",
        "primary_use_case": "Standard tabular machine learning benchmark for operational process prediction (e.g. Did Not Attend risk) and clinical outcome classification.",
        "loss_function": r"\hat{P}(Y = 1 | X) = \frac{1}{B} \sum_{b=1}^B T_b(X)",
        "loss_explanation": "Averages probability outputs across B independently trained trees. Bagging reduces model variance by a factor approaching 1/B while preserving the low bias of deep individual trees.",
        "hyperparameters": [
            {"param": "n_estimators", "name": "Number of Trees (B)", "default": "200", "range": "50 - 500", "description": "Higher values monotonically stabilize ensemble variance with diminishing returns."},
            {"param": "max_features", "name": "Feature Sub-sampling Ratio", "default": "'sqrt'", "range": "sqrt(p) or log2(p)", "description": "Forces trees to evaluate alternative features, decorrelating individual tree errors."}
        ],
        "pitfalls": [
            "Default threshold of 0.5 collapses in imbalanced settings; probability calibration or threshold tuning is essential.",
            "Feature importance can be biased towards continuous variables with many unique values."
        ],
        "pros": ["Excellent out-of-the-box accuracy", "No feature scaling needed", "Resistant to overfitting compared to single trees"],
        "cons": ["Ensemble is a black-box requiring SHAP for explanation", "Slower inference than linear models"]
    },

    "gradient_boosting_classifier": {
        "id": "gradient_boosting_classifier",
        "name": "Gradient Boosted Decision Trees (XGBoost / LightGBM)",
        "family": "classification",
        "short_desc": "Sequential boosting ensemble that trains shallow decision trees iteratively, each correcting the residual pseudo-errors of previous iterations.",
        "interpretability": "Moderate to Low",
        "interpretability_score": 2,
        "sample_efficiency": "Moderate to High",
        "handles_non_linear": True,
        "handles_imbalance": "Via Positive Class Scale Weight (`scale_pos_weight`)",
        "primary_use_case": "Top-tier tabular performance for complex operational forecasting (ED breach risk, readmissions) where predictive accuracy is paramount.",
        "loss_function": r"\mathcal{L}^{(t)} = \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2",
        "loss_explanation": "Second-order Taylor expansion of the loss function, where g_i and h_i are first and second order gradients of log-loss. Regularizes tree complexity (T leaves, L2 leaf weights w).",
        "hyperparameters": [
            {"param": "learning_rate", "name": "Shrinkage Rate (η)", "default": "0.05", "range": "0.01 - 0.20", "description": "Shrinks tree contributions to prevent premature convergence."},
            {"param": "n_estimators", "name": "Boosting Rounds", "default": "150", "range": "50 - 500", "description": "Number of sequential trees; tuned with early stopping on validation loss."},
            {"param": "max_depth", "name": "Tree Depth", "default": "3", "range": "2 - 6", "description": "Shallow trees (stumps or depth 3-4) limit higher-order interaction complexity."}
        ],
        "pitfalls": [
            "Can overfit rapidly if learning rate is too high or early stopping is omitted on noisy clinical data.",
            "Requires careful hyperparameter tuning compared to Random Forest."
        ],
        "pros": ["State-of-the-art accuracy on tabular health datasets", "Native handling of missing values", "Built-in regularization"],
        "cons": ["Requires hyperparameter optimization", "Opaque decision surface requiring SHAP analysis"]
    },

    "svm_classifier": {
        "id": "svm_classifier",
        "name": "Support Vector Machine (Linear / RBF Kernel)",
        "family": "classification",
        "short_desc": "Maximizes the geometric margin separating classes in an implicit high-dimensional reproducing kernel Hilbert space.",
        "interpretability": "Low",
        "interpretability_score": 2,
        "sample_efficiency": "High for high-dimensional data (p > N)",
        "handles_non_linear": True,
        "handles_imbalance": "Via Class Weights (`class_weight='balanced'`)",
        "primary_use_case": "High-dimensional genomics, transcriptomics, or spectral metabolomics where feature count exceeds patient count.",
        "loss_function": r"\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i (w^T \phi(x_i) + b) \ge 1 - \xi_i",
        "loss_explanation": "Minimizes the soft-margin hinge loss. Parameter C balances maximizing the separation margin with penalizing classification violations (slack variables xi). RBF kernel maps features into infinite dimensions.",
        "hyperparameters": [
            {"param": "C", "name": "Soft-Margin Penalty", "default": "1.0", "range": "0.1 - 100.0", "description": "Higher values strictly penalize margin errors; lower values widen the margin."},
            {"param": "gamma", "name": "RBF Kernel Bandwidth", "default": "'scale'", "range": "0.001 - 1.0", "description": "Defines the radius of influence of individual support vectors."}
        ],
        "pitfalls": [
            "Severely degraded performance if continuous features are not strictly normalized/standardized.",
            "Computationally quadratic to cubic O(N^2 to N^3) with respect to sample size, scaling poorly beyond 50,000 samples."
        ],
        "pros": ["Highly robust in genomics (p >> N)", "Global optimum guaranteed (convex optimization)"],
        "cons": ["Does not produce native probability estimates without Platt scaling", "Computationally slow on large datasets"]
    },

    "knn_classifier": {
        "id": "knn_classifier",
        "name": "k-Nearest Neighbors (k-NN Classifier)",
        "family": "classification",
        "short_desc": "Instance-based non-parametric classifier assigning class membership via majority vote among the k closest training observations in Euclidean space.",
        "interpretability": "Moderate",
        "interpretability_score": 3,
        "sample_efficiency": "Low to Moderate",
        "handles_non_linear": True,
        "handles_imbalance": "Poor (biased towards majority class without distance weighting)",
        "primary_use_case": "Intuitive peer-matching and case-based retrieval (e.g. 'find the 5 most similar patients who presented with these symptoms').",
        "loss_function": r"\hat{P}(Y = c | X) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(X)} \mathbb{I}(y_i = c)",
        "loss_explanation": "Identifies the k nearest neighbors using Euclidean, Manhattan, or Mahalanobis distance, computing posterior class probability as the local neighborhood proportion.",
        "hyperparameters": [
            {"param": "n_neighbors", "name": "Number of Neighbors (k)", "default": "7", "range": "3 - 25", "description": "Smaller k yields complex localized boundaries (high variance); larger k smooths boundaries (high bias)."},
            {"param": "weights", "name": "Distance Weighting", "default": "'distance'", "options": "['uniform', 'distance']", "description": "Weights neighbor votes by inverse Euclidean distance."}
        ],
        "pitfalls": [
            "Catastrophically sensitive to unscaled features and the 'curse of dimensionality' (distances become equidistant as p increases).",
            "Slow inference: requires scanning entire training database for every prediction."
        ],
        "pros": ["Zero training phase", "Intuitive clinical concept of 'similar historical cases'"],
        "cons": ["Requires aggressive dimensionality reduction", "High memory and inference cost"]
    },

    # --- Regression Models ---
    "elastic_net_regression": {
        "id": "elastic_net_regression",
        "name": "ElasticNet Penalized Regression (L1 + L2)",
        "family": "regression",
        "short_desc": "Linear regression combining L1 (Lasso) and L2 (Ridge) penalties to handle multicollinear continuous predictors and perform sparse variable selection.",
        "interpretability": "High",
        "interpretability_score": 5,
        "sample_efficiency": "High (effective for p > N)",
        "handles_non_linear": False,
        "handles_imbalance": "N/A",
        "primary_use_case": "Predicting hospital length of stay, medical expenditures, or physiological biomarkers when clinical predictors are highly correlated.",
        "loss_function": r"\mathcal{L}(\beta) = \frac{1}{2n} \sum_{i=1}^n (y_i - x_i^T \beta)^2 + \lambda \left( \alpha \|\beta\|_1 + \frac{1 - \alpha}{2} \|\beta\|_2^2 \right)",
        "loss_explanation": "Ordinary least squares residual sum of squares augmented with an ElasticNet penalty. Overcomes Lasso's limitation of selecting only one variable from a group of correlated predictors by borrowing Ridge's grouping property.",
        "hyperparameters": [
            {"param": "alpha", "name": "Overall Regularization Strength", "default": "0.1", "range": "0.001 - 10.0", "description": "Higher values shrink coefficients closer to zero."},
            {"param": "l1_ratio", "name": "L1 / L2 Balance", "default": "0.5", "range": "0.0 - 1.0", "description": "0.0 is pure Ridge; 1.0 is pure Lasso."}
        ],
        "pitfalls": [
            "Cannot capture non-linear ceiling effects or threshold interactions without polynomial terms.",
            "Predictors must be centered and scaled before fitting."
        ],
        "pros": ["Solves multicollinearity cleanly", "Automates feature selection", "Interpretable linear slope coefficients"],
        "cons": ["Biased coefficient estimates due to shrinkage", "Assumes strictly linear relationship"]
    },

    "random_forest_regressor": {
        "id": "random_forest_regressor",
        "name": "Random Forest Regressor (Ensemble)",
        "family": "regression",
        "short_desc": "Ensemble of unpruned regression trees aggregating continuous predictions across bootstrap samples to model non-linear physiological interactions.",
        "interpretability": "Moderate",
        "interpretability_score": 3,
        "sample_efficiency": "Moderate to High",
        "handles_non_linear": True,
        "handles_imbalance": "N/A",
        "primary_use_case": "Predicting surgical case duration, elective theatre turnaround times, or non-linear clinical biomarker titration.",
        "loss_function": r"\hat{y}(x) = \frac{1}{B} \sum_{b=1}^B T_b(x) \quad \text{where} \quad \text{MSE}_{\text{node}} = \frac{1}{N} \sum_{i \in \text{node}} (y_i - \bar{y})^2",
        "loss_explanation": "Splits each node to minimize variance (mean squared error), then averages predictions across all B trees to produce smooth continuous response surfaces.",
        "hyperparameters": [
            {"param": "n_estimators", "name": "Number of Trees", "default": "200", "range": "50 - 500", "description": "Number of trees in ensemble."},
            {"param": "max_depth", "name": "Max Depth", "default": "8", "range": "3 - 20", "description": "Limits tree depth to prevent memorizing continuous training noise."}
        ],
        "pitfalls": [
            "Cannot extrapolate beyond the minimum and maximum values observed in the training response data.",
            "Can produce stepped, non-smooth continuous predictions."
        ],
        "pros": ["Captures complex non-linear clinical thresholds", "Robust to extreme outliers in features"],
        "cons": ["Cannot extrapolate trends", "Black-box ensemble"]
    },

    "gradient_boosting_regressor": {
        "id": "gradient_boosting_regressor",
        "name": "Gradient Boosted Regressor (XGBoost / LightGBM)",
        "family": "regression",
        "short_desc": "Sequential additive regression ensemble fitting shallow trees to the negative gradient (residuals) of continuous loss functions.",
        "interpretability": "Moderate to Low",
        "interpretability_score": 2,
        "sample_efficiency": "High",
        "handles_non_linear": True,
        "handles_imbalance": "N/A",
        "primary_use_case": "High-accuracy operational prediction of elective surgical durations, bed occupancy hours, and multi-variable clinical trajectories.",
        "loss_function": r"\mathcal{L}_{\text{Huber}}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}",
        "loss_explanation": "Optimizes MSE, MAE, or robust Huber loss. Huber loss behaves quadratically for small residuals and linearly for large outliers, making it robust against clinical recording errors.",
        "hyperparameters": [
            {"param": "learning_rate", "name": "Learning Rate", "default": "0.05", "range": "0.01 - 0.20", "description": "Shrinks tree contributions to control variance."},
            {"param": "loss", "name": "Loss Criterion", "default": "'huber'", "options": "['squared_error', 'absolute_error', 'huber']", "description": "Huber is robust to extreme clinical outlier values."}
        ],
        "pitfalls": [
            "Prone to overfitting on small clinical trial cohorts if boosting rounds are not regularized.",
            "Sensitive to hyperparameter settings."
        ],
        "pros": ["Consistently highest predictive accuracy on tabular regression", "Flexible robust loss functions"],
        "cons": ["Requires hyperparameter tuning", "Black-box model"]
    },

    # --- Time Series Forecasting ---
    "holt_winters_exponential_smoothing": {
        "id": "holt_winters_exponential_smoothing",
        "name": "Triple Exponential Smoothing (Holt-Winters)",
        "family": "time_series",
        "short_desc": "Classical statistical forecasting decomposing sequential data into exponentially weighted Level, Trend, and Seasonal components.",
        "interpretability": "High",
        "interpretability_score": 5,
        "sample_efficiency": "Very High (effective with 1-2 years of historical data)",
        "handles_non_linear": False,
        "handles_imbalance": "N/A",
        "primary_use_case": "Forecasting daily Emergency Department (ED) attendances, weekly clinic capacity demand, and bed occupancy.",
        "loss_function": r"\hat{y}_{t+h|t} = \ell_t + h b_t + s_{t+h-m(k+1)} \quad \text{where} \quad \ell_t = \alpha(y_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1})",
        "loss_explanation": "Recursively updates Level (alpha), Trend (beta), and Seasonal indices (gamma). Generates forward multi-step forecasts with analytical confidence intervals based on recursive forecast error variance.",
        "hyperparameters": [
            {"param": "seasonal_periods", "name": "Seasonal Frequency (m)", "default": "7", "options": "[7 (Daily/Weekly cycle), 52 (Weekly/Annual cycle)]", "description": "The number of periods in a complete seasonal cycle (7 for day-of-week)."},
            {"param": "seasonal", "name": "Seasonality Type", "default": "'add'", "options": "['add', 'mul']", "description": "Additive if seasonal swings are constant; Multiplicative if swings grow with level."}
        ],
        "pitfalls": [
            "Cannot easily incorporate external exogenous covariates (e.g. extreme heatwaves, bank holidays) without SARIMAX.",
            "Requires continuous sequential temporal data without missing calendar dates."
        ],
        "pros": ["Transparent decomposition (Level, Trend, Seasonality)", "Fast calculation and intuitive parameters", "Closed-form prediction intervals"],
        "cons": ["Assumes regular, repeating seasonal cycles", "Sensitive to sudden structural breaks"]
    },

    "sarima_forecasting": {
        "id": "sarima_forecasting",
        "name": "Seasonal ARIMA (SARIMA / SARIMAX)",
        "family": "time_series",
        "short_desc": "Autoregressive Integrated Moving Average modeling stationary differenced series, seasonal lag dependencies, and exogenous operational shocks.",
        "interpretability": "Moderate",
        "interpretability_score": 4,
        "sample_efficiency": "Moderate",
        "handles_non_linear": False,
        "handles_imbalance": "N/A",
        "primary_use_case": "Medium-term hospital demand modeling, multi-month bed capacity planning, and regional epidemiological monitoring.",
        "loss_function": r"\Phi_P(B^s)\phi_p(B)(1 - B)^d (1 - B^s)^D y_t = \Theta_Q(B^s)\theta_q(B)\varepsilon_t",
        "loss_explanation": "Combines non-seasonal (p, d, q) and seasonal (P, D, Q)_s operators with backshift polynomial B. Minimizes Gaussian log-likelihood via maximum likelihood estimation.",
        "hyperparameters": [
            {"param": "order", "name": "ARIMA Order (p, d, q)", "default": "(1, 1, 1)", "description": "Autoregressive terms (p), differencing degree (d), moving average terms (q)."},
            {"param": "seasonal_order", "name": "Seasonal Order (P, D, Q, s)", "default": "(1, 1, 1, 7)", "description": "Seasonal lag terms and cycle length s."}
        ],
        "pitfalls": [
            "Series must be made stationary via differencing (Augmented Dickey-Fuller test prerequisite).",
            "Complex parameter identification (AIC/BIC selection) can be daunting for non-statisticians."
        ],
        "pros": ["Theoretically rigorous Box-Jenkins methodology", "Supports exogenous predictors (SARIMAX)", "Accurate prediction intervals"],
        "cons": ["Requires stationary differencing", "Computationally slower than Holt-Winters"]
    },

    "lagged_ensemble_forecasting": {
        "id": "lagged_ensemble_forecasting",
        "name": "Lagged Feature Supervised Tree Ensemble",
        "family": "time_series",
        "short_desc": "Transforms time series forecasting into a tabular supervised learning problem using rolling window aggregations, lag features, and calendar indicators.",
        "interpretability": "Moderate",
        "interpretability_score": 3,
        "sample_efficiency": "Moderate to High",
        "handles_non_linear": True,
        "handles_imbalance": "N/A",
        "primary_use_case": "Complex operational forecasting incorporating non-linear weather features, public holidays, staff rotas, and multi-channel demand signals.",
        "loss_function": r"\hat{y}_{t+1} = f(y_t, y_{t-1}, y_{t-7}, \text{RollingMean}_{7}(y), \text{DayOfWeek}_t, \text{BankHoliday}_t)",
        "loss_explanation": "Constructs a feature matrix of historical lags and rolling summary statistics, then trains a Gradient Boosted Regressor to map past states to forward demand.",
        "hyperparameters": [
            {"param": "max_lag", "name": "Maximum Historical Lag (Days)", "default": "14", "range": "7 - 28", "description": "Number of sequential lagged days included as features."},
            {"param": "rolling_windows", "name": "Rolling Summary Windows", "default": "[7, 14]", "description": "Rolling mean and standard deviation windows."}
        ],
        "pitfalls": [
            "CRITICAL: Must NEVER use standard K-Fold shuffle cross-validation! Must strictly use TimeSeriesSplit / Expanding Window cross-validation to prevent temporal data leakage.",
            "Requires recursive multi-step forecasting for horizons beyond 1 step ahead."
        ],
        "pros": ["Naturally accommodates exogenous operational covariates", "Captures non-linear holiday spikes", "High predictive power"],
        "cons": ["Risk of future-to-past data leakage if validation is poorly designed", "No closed-form prediction intervals"]
    },

    # --- Unsupervised Clustering & Dimensionality Reduction ---
    "kmeans_clustering": {
        "id": "kmeans_clustering",
        "name": "K-Means Clustering",
        "family": "clustering",
        "short_desc": "Partitions observations into k non-overlapping clusters by iteratively assigning points to the nearest cluster centroid and updating centroids.",
        "interpretability": "High",
        "interpretability_score": 4,
        "sample_efficiency": "High (scales linearly with N)",
        "handles_non_linear": False,
        "handles_imbalance": "N/A",
        "primary_use_case": "Patient phenotyping, clinical service utilization segmentation, and hospital ward workload grouping.",
        "loss_function": r"J = \sum_{k=1}^K \sum_{i \in C_k} \|x_i - \mu_k\|^2",
        "loss_explanation": "Minimizes the Within-Cluster Sum of Squares (WCSS / Inertia). Lloyd's algorithm alternates between assigning points to the nearest centroid and recomputing centroids as cluster means.",
        "hyperparameters": [
            {"param": "n_clusters", "name": "Number of Clusters (k)", "default": "3", "range": "2 - 8", "description": "Selected via the Elbow Method (inflection point of WCSS) and Silhouette Score."},
            {"param": "init", "name": "Centroid Initialization", "default": "'k-means++'", "options": "['k-means++', 'random']", "description": "k-means++ spreads initial seeds to avoid suboptimal local minima."}
        ],
        "pitfalls": [
            "Assumes spherical, equal-variance clusters; performs poorly on elongated, curved, or varying-density clusters.",
            "Forced assignment: every point is forced into a cluster even if it is a severe outlier or noise."
        ],
        "pros": ["Fast, scalable O(k*N*p) runtime", "Intuitive centroid summaries", "Clear mathematical interpretation"],
        "cons": ["Must specify k in advance", "Sensitive to outliers and feature scaling", "Fails on complex spatial topologies"]
    },

    "dbscan_clustering": {
        "id": "dbscan_clustering",
        "name": "DBSCAN (Density-Based Spatial Clustering with Noise)",
        "family": "clustering",
        "short_desc": "Density-based clustering that discovers arbitrary non-linear cluster shapes and isolates sparse observations as background noise.",
        "interpretability": "Moderate",
        "interpretability_score": 3,
        "sample_efficiency": "Moderate",
        "handles_non_linear": True,
        "handles_imbalance": "N/A",
        "primary_use_case": "Public health outbreak hotspot detection, bacterial genomic isolate clustering, and identifying anomalous clinical outliers.",
        "loss_function": r"N_\varepsilon(p) = \{q \in D \mid \text{dist}(p, q) \le \varepsilon\} \quad \text{Core point if } |N_\varepsilon(p)| \ge \text{MinPts}",
        "loss_explanation": "Grows clusters from dense core points having at least MinPts neighbors within radius epsilon. Points that are neither core points nor reachable are tagged as noise (-1).",
        "hyperparameters": [
            {"param": "eps", "name": "Neighborhood Radius (ε)", "default": "0.5", "range": "0.1 - 2.0", "description": "Maximum distance between two points to be considered neighbors; tuned via k-distance graph."},
            {"param": "min_samples", "name": "Minimum Points (MinPts)", "default": "5", "range": "3 - 20", "description": "Minimum points required to form a dense core region (rule-of-thumb: >= 2 * features)."}
        ],
        "pitfalls": [
            "Cannot handle clusters with widely differing densities, as a single epsilon radius cannot capture both tight and loose clusters.",
            "Distance metrics degrade in high dimensions."
        ],
        "pros": ["Does NOT require specifying number of clusters k", "Discovers arbitrary non-linear and crescent shapes", "Robust outlier / noise rejection"],
        "cons": ["Sensitive to epsilon choice", "Struggles with varying density clusters"]
    },

    "pca_reduction": {
        "id": "pca_reduction",
        "name": "Principal Component Analysis (PCA)",
        "family": "dim_reduction",
        "short_desc": "Linear transformation projecting correlated continuous features onto orthogonal principal axes that maximize captured variance.",
        "interpretability": "Moderate",
        "interpretability_score": 3,
        "sample_efficiency": "High",
        "handles_non_linear": False,
        "handles_imbalance": "N/A",
        "primary_use_case": "Dimensionality reduction for genomics/transcriptomics, visualizing high-dimensional patient cohorts in 2D, and eliminating multicollinearity.",
        "loss_function": r"\Sigma = \frac{1}{n} X^T X = V \Lambda V^T \quad \text{where} \quad z_1 = X v_1 \text{ maximizes } \text{Var}(z_1)",
        "loss_explanation": "Computes the spectral eigen-decomposition of the sample covariance matrix (or SVD of data matrix X). The eigenvectors V form the principal loading axes, and eigenvalues Lambda reflect variance explained.",
        "hyperparameters": [
            {"param": "n_components", "name": "Retained Components", "default": "2", "range": "2 - 10", "description": "Number of principal axes to retain; chosen to capture >= 70-80% cumulative variance."},
            {"param": "whiten", "name": "Whitening", "default": "False", "options": "[True, False]", "description": "Removes linear correlations and forces component variances to 1."}
        ],
        "pitfalls": [
            "Data must be strictly centered and standardized (z-scored); otherwise, features with large numeric scales dominate the principal components.",
            "Linear only: fails to capture non-linear manifold structures (e.g. Swiss roll)."
        ],
        "pros": ["Guaranteed orthogonal decorrelated components", "Visualizes high-dimensional data in 2D/3D", "Identifies dominant patterns via loadings"],
        "cons": ["Components are linear combinations, reducing direct clinical interpretability", "Sensitive to extreme outliers"]
    },

    # --- Algorithmic Fairness & Equity Audit ---
    "algorithmic_fairness_audit": {
        "id": "algorithmic_fairness_audit",
        "name": "Algorithmic Fairness & Health Equity Disparity Audit",
        "family": "fairness_audit",
        "short_desc": "Auditing operational and clinical machine learning models to detect, measure, and mitigate demographic bias across protected groups (IMD deprivation, ethnicity, age).",
        "interpretability": "Very High",
        "interpretability_score": 5,
        "sample_efficiency": "Requires adequate subgroup representation",
        "handles_non_linear": True,
        "handles_imbalance": "Explicitly addresses equity in minority subgroups",
        "primary_use_case": "Vetting clinical risk stratification, outpatient appointment scheduling (DNA prediction), and resource allocation algorithms for discriminatory impact.",
        "loss_function": r"\text{Disparate Impact} = \frac{P(\hat{Y}=1 | A=\text{unprivileged})}{P(\hat{Y}=1 | A=\text{privileged})} \ge 0.80 \quad \text{(Four-Fifths Rule)}",
        "loss_explanation": "Audits selection rates and error distributions across protected groups A. Measures Demographic Parity Difference, Disparate Impact Ratio, and Equalized Odds (False Positive Rate parity and True Positive Rate parity).",
        "hyperparameters": [
            {"param": "protected_attribute", "name": "Sensitive Attribute", "default": "'IMD_Decile'", "description": "The equity dimension being audited (e.g. Index of Multiple Deprivation quintiles, Ethnicity, Age band)."},
            {"param": "fairness_criterion", "name": "Fairness Target", "default": "'equalized_odds'", "options": "['demographic_parity', 'equalized_odds', 'disparate_impact']", "description": "Demographic parity enforces equal selection rates; Equalized odds enforces equal sensitivity and specificity across groups."}
        ],
        "pitfalls": [
            "Fairness through Unawareness: Merely dropping protected attributes (e.g. ethnicity) does NOT eliminate bias because proxy variables (postcode, insurance, comorbidities) encode the disparity.",
            "Impossibility Theorem of Machine Learning Fairness: In general, Demographic Parity, Equalized Odds, and Predictive Value Parity cannot all be satisfied simultaneously unless base prevalence rates are identical."
        ],
        "pros": ["Essential for NHS AI ethics and regulatory compliance", "Exposes hidden algorithmic discrimination before deployment", "Protects vulnerable and deprived patient populations"],
        "cons": ["Mathematical tension between competing fairness criteria", "Requires granular demographic data collection"]
    }
}

# Ensure all models have alias 'description' populated from 'short_desc'
for _model in ML_MODELS.values():
    if "description" not in _model:
        _model["description"] = _model.get("short_desc", "")


# =====================================================================
# 3. Decision Wizard Routing Tree Schema
# =====================================================================

ML_DECISION_NODES: List[Dict[str, Any]] = [
    {
        "id": "task_type",
        "question": "What is the primary analytical or operational objective of your machine learning project?",
        "options": [
            {"value": "classification", "label": "Predicting Categorical Clinical Event or Operational Risk (e.g. Readmission, Did Not Attend)", "next": "class_imbalance"},
            {"value": "regression", "label": "Predicting a Continuous Quantity or Clinical Score (e.g. Length of Stay, Recovery Time)", "next": "reg_linearity"},
            {"value": "time_series", "label": "Forward Demand & Capacity Forecasting over Time (e.g. Daily A&E Attendances, Sickness)", "next": "ts_frequency"},
            {"value": "clustering", "label": "Discovering Latent Patient Subgroups, Phenotypes, or Outbreak Clusters (Unsupervised)", "next": "clust_shape"},
            {"value": "dim_reduction", "label": "Compressing High-Dimensional Genomic or Laboratory Data into 2D/Summary Axes", "next": "dim_linearity"},
            {"value": "fairness_audit", "label": "Auditing an Existing Clinical/Operational Algorithm for Bias & Health Equity Disparities", "next": "fairness_metric"}
        ]
    },
    {
        "id": "class_imbalance",
        "question": "What is the incidence / event rate of the target outcome in your patient cohort?",
        "options": [
            {"value": "severe_imbalance", "label": "Rare Event / Severe Imbalance (< 10% positive cases, e.g. 30-day readmission)", "next": "class_interpretability"},
            {"value": "moderate_imbalance", "label": "Moderate Imbalance (10% to 30% positive cases, e.g. Outpatient DNA)", "next": "class_interpretability"},
            {"value": "balanced", "label": "Roughly Balanced (30% to 50% positive cases)", "next": "class_interpretability"}
        ]
    },
    {
        "id": "class_interpretability",
        "question": "What is the priority between model transparency (clinician trust) vs. maximal predictive power?",
        "options": [
            {"value": "high_interpretability", "label": "Strict Transparency Required: Must explain exact risk factors (Odds Ratios or Bedside Tree Rules)", "recommend": "penalized_logistic_regression"},
            {"value": "balanced", "label": "Balanced: Strong predictive power with feature importance rankings & SHAP explanation", "recommend": "random_forest_classifier"},
            {"value": "max_accuracy", "label": "Maximum Accuracy: Willing to accept a complex ensemble for top-tier predictive discrimination", "recommend": "gradient_boosting_classifier"},
            {"value": "high_dimensions", "label": "High-Dimensional Data: Genomic or biomarker panel where features exceed patients (p > N)", "recommend": "svm_classifier"}
        ]
    },
    {
        "id": "reg_linearity",
        "question": "What is the underlying relationship between clinical predictors and your continuous outcome?",
        "options": [
            {"value": "linear", "label": "Primarily Linear or Multicollinear Predictors: Need stable slope coefficients and feature selection", "recommend": "elastic_net_regression"},
            {"value": "non_linear_robust", "label": "Complex Non-Linear Thresholds: Prone to clinical outliers, non-smooth response surfaces", "recommend": "random_forest_regressor"},
            {"value": "max_precision", "label": "Top Precision Tabular Regression: Demanding lowest possible MAE/RMSE on operational metrics", "recommend": "gradient_boosting_regressor"}
        ]
    },
    {
        "id": "ts_frequency",
        "question": "What is the temporal frequency and structure of your operational time series data?",
        "options": [
            {"value": "daily_weekly_cycle", "label": "Daily Attendances with Strong Weekly Day-of-Week Rhythm (e.g. A&E Monday peaks)", "recommend": "holt_winters_exponential_smoothing"},
            {"value": "seasonal_autoregressive", "label": "Multi-Month / Weekly Trends with Autoregressive Lags (e.g. Staff Sickness Waves)", "recommend": "sarima_forecasting"},
            {"value": "multi_feature_covariates", "label": "Demand with External Factors: Weather, Public Holidays, Staffing Rotas (Supervised)", "recommend": "lagged_ensemble_forecasting"}
        ]
    },
    {
        "id": "clust_shape",
        "question": "What topological cluster structure do you anticipate in your clinical or geographical data?",
        "options": [
            {"value": "spherical", "label": "Compact, Spherical Clusters: Clear patient subgroups around central clinical prototypes", "recommend": "kmeans_clustering"},
            {"value": "arbitrary_density", "label": "Arbitrary Non-Linear Shapes or Spatial Outbreak Density with Background Noise", "recommend": "dbscan_clustering"}
        ]
    },
    {
        "id": "dim_linearity",
        "question": "What is your primary goal for dimensionality reduction?",
        "options": [
            {"value": "linear_variance", "label": "Linear Variance Compression: Eliminate multicollinearity and identify dominant orthogonal loading axes", "recommend": "pca_reduction"}
        ]
    },
    {
        "id": "fairness_metric",
        "question": "What is the primary equity concern regarding algorithm deployment?",
        "options": [
            {"value": "disparate_impact", "label": "Disparate Impact & Error Parity across Deprivation Deciles (IMD) or Demographic Groups", "recommend": "algorithmic_fairness_audit"}
        ]
    }
]


# =====================================================================
# 4. Realistic Fictional UK Healthcare Clinical & Operational Scenarios
# =====================================================================

ML_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "clinical_readmission": {
        "scenario_key": "clinical_readmission",
        "name": "All Meadows Cardiovascular Institute: 30-Day Emergency Heart Failure Readmission Risk",
        "organisation_name": "All Meadows Cardiovascular Institute",
        "fictional_trust": "All Meadows Cardiovascular Institute",
        "organisation": "All Meadows Cardiovascular Institute (Clinical Trials & Predictive Analytics Unit)",
        "domain": "Cardiovascular Medicine, Prognostic Risk Stratification & Clinical Decision Support",
        "task_family": "classification",
        "recommended_model": "random_forest_classifier",
        "comparator_model": "penalized_logistic_regression",
        "primary_objective": "Predicting binary 30-day emergency hospital readmission following acute decompensated heart failure discharge.",
        "target_variable": "Readmitted_30d (Binary: 1 = Readmitted, 0 = Safely Discharged)",
        "imbalance_note": "Severe 8.0% event rate (92% safe discharge). Standard accuracy is deceptive; model performance must be evaluated via PR-AUC and F1-macro.",
        "clinical_features": [
            "Age_Years", "Baseline_LVEF_Pct", "Prior_Admissions_12m", "Serum_Creatinine_umolL",
            "Systolic_BP_mmHg", "Beta_Blocker_Prescribed", "Diabetes_Mellitus", "NYHA_Class"
        ],
        "description": "Double-blind predictive model evaluating post-discharge clinical risk to target early nurse-led virtual ward follow-up.",
        "clinical_context": "Synthetic clinical observational cohort simulating adult heart failure discharges across a tertiary NHS cardiac centre.",
        "takeaway": "When predicting rare clinical outcomes (8% readmission), an uncalibrated classifier predicting 100% negative achieves 92% accuracy while being clinically useless. Random Forest with balanced sub-sampling captures non-linear LVEF/Creatinine interactions, yielding an optimal PR-AUC of 0.48 (6x baseline chance).",
        "is_synthetic": True,
        "disclaimer": "100% synthetic artificial dataset generated purely for educational and machine learning demonstration. Strictly fictional UK healthcare institution."
    },

    "operational_dna_equity": {
        "scenario_key": "operational_dna_equity",
        "name": "Riverside Community Health Trust: Outpatient 'Did Not Attend' (DNA) Risk & Algorithmic Equity Audit",
        "organisation_name": "Riverside Community Health Trust",
        "fictional_trust": "Riverside Community Health Trust",
        "organisation": "Riverside Community Health Trust (Digital Health & Health Equity Unit)",
        "domain": "Hospital Operational Efficiency, Outpatient Access & Algorithmic Fairness",
        "task_family": "fairness_audit",
        "recommended_model": "algorithmic_fairness_audit",
        "comparator_model": "penalized_logistic_regression",
        "primary_objective": "Predicting outpatient missed appointment (DNA) probability while auditing algorithmic disparity across Index of Multiple Deprivation (IMD) deciles.",
        "target_variable": "Did_Not_Attend (Binary: 1 = DNA, 0 = Attended)",
        "protected_attribute": "IMD_Deprivation_Tier (High Deprivation Deciles 1-3 vs. Moderate Deciles 4-7 vs. Low Deciles 8-10)",
        "clinical_features": [
            "Patient_Age", "IMD_Decile", "Travel_Distance_Miles", "Prior_DNA_Count_24m",
            "Appointment_Lead_Days", "Clinical_Specialty", "SMS_Reminder_Delivered", "Public_Transport_Time_Mins"
        ],
        "description": "Operational scheduling algorithm designed to offer transport vouchers and proactive telephone support to high-risk patients, audited to prevent discriminatory double-booking.",
        "clinical_context": "Synthetic operational dataset evaluating outpatient attendance patterns and socio-economic deprivation gradients.",
        "takeaway": "Using appointment DNA risk to double-book clinic slots severely penalizes deprived populations: patients in IMD Deciles 1-3 have a 2.4x higher false-positive rate. Auditing Disparate Impact and enforcing False Positive Parity ensures supportive interventions (transport aid) are allocated equitably without worsening clinic wait times for vulnerable groups.",
        "is_synthetic": True,
        "disclaimer": "100% synthetic artificial dataset generated purely for educational demonstration. Strictly fictional UK healthcare institution."
    },

    "ed_demand_forecasting": {
        "scenario_key": "ed_demand_forecasting",
        "name": "Inner City Teaching Hospital: Daily Emergency Department (ED) Demand & Bed Surge Forecasting",
        "organisation_name": "Inner City Teaching Hospital",
        "fictional_trust": "Inner City Teaching Hospital",
        "organisation": "Inner City Teaching Hospital (Emergency Medicine & Operational Informatics Collaborative)",
        "domain": "Urgent & Emergency Care, Hospital Operations & Bed Capacity Planning",
        "task_family": "time_series",
        "recommended_model": "holt_winters_exponential_smoothing",
        "comparator_model": "lagged_ensemble_forecasting",
        "primary_objective": "Forecasting daily A&E patient arrival volumes and associated inpatient bed requirements over a 14-day forward planning horizon.",
        "target_variable": "Daily_ED_Attendances (Continuous Daily Count)",
        "temporal_structure": "365 daily observations exhibiting a pronounced 7-day day-of-week rhythm (Monday surge) and winter respiratory volume spikes.",
        "clinical_features": [
            "Date", "Day_of_Week", "Is_Weekend", "Winter_Surge_Season", "Previous_Day_Attendances",
            "Rolling_7d_Mean_Attendances", "Emergency_Admissions_Count"
        ],
        "description": "Multi-period capacity planning model used by the Hospital Bed Management Team to dynamically adjust nursing rotas and elective surgery allocations.",
        "clinical_context": "Synthetic time series of daily Emergency Department arrivals with day-of-week seasonality and winter surge dynamics.",
        "takeaway": "Emergency Department attendance follows a rigid 7-day cyclical rhythm with Monday attendances exceeding weekend volumes by 28%. Triple Exponential Smoothing (Holt-Winters) accurately captures weekly seasonality and winter trend drift, yielding an out-of-sample MAE of +/- 9 patients/day with valid 95% prediction intervals.",
        "is_synthetic": True,
        "disclaimer": "100% synthetic artificial dataset generated purely for educational demonstration. Strictly fictional UK healthcare institution."
    },

    "workforce_sickness_forecasting": {
        "scenario_key": "workforce_sickness_forecasting",
        "name": "NHS Integrated Care Board: Frontline Nursing Sickness Absence & Rotas Depletion Forecasting",
        "organisation_name": "NHS Integrated Care Board",
        "fictional_trust": "NHS Integrated Care Board",
        "organisation": "NHS Integrated Care Board (Workforce Intelligence & Operational Planning Directorate)",
        "domain": "Health Workforce Analytics, Staff Wellbeing & Rota Resilience",
        "task_family": "time_series",
        "recommended_model": "sarima_forecasting",
        "comparator_model": "holt_winters_exponential_smoothing",
        "primary_objective": "Forecasting weekly staff sickness absence percentage (%) across hospital trusts to plan temporary bank and agency nursing buffer capacity.",
        "target_variable": "Sickness_Absence_Pct (Continuous Percentage: 3.5% to 8.5%)",
        "temporal_structure": "52 sequential weekly observations capturing annual respiratory surge and mid-winter staffing troughs.",
        "clinical_features": [
            "Week_Number", "Season", "Sickness_Absence_Pct", "Full_Time_Equivalent_Staff",
            "Regional_Influenza_Positivity_Pct", "Bank_Staff_Fill_Rate_Pct"
        ],
        "description": "Regional workforce predictive model anticipating clinical staffing deficits 4 to 8 weeks in advance of winter respiratory peaks.",
        "clinical_context": "Synthetic longitudinal workforce metrics tracking clinical absenteeism and seasonal respiratory virus waves.",
        "takeaway": "Staff sickness rates surge from a baseline of 4.2% in summer to over 7.8% in December/January. Seasonal Autoregressive models (SARIMA) effectively model 4-week lag autocorrelation, enabling the trust to pre-book bank clinical staff 6 weeks prior to anticipated winter capacity crunches.",
        "is_synthetic": True,
        "disclaimer": "100% synthetic artificial dataset generated purely for educational demonstration. Strictly fictional UK healthcare institution."
    },

    "manufacturing_predictive_maintenance": {
        "scenario_key": "manufacturing_predictive_maintenance",
        "name": "Apex Precision Manufacturing: Industrial Equipment Spindle Failure & Predictive Maintenance",
        "organisation_name": "Apex Precision Robotics & Advanced Manufacturing",
        "fictional_trust": "Apex Precision Robotics & Advanced Manufacturing",
        "organisation": "Apex Precision Robotics (Industry 4.0 Reliability Engineering & IoT Diagnostics)",
        "domain": "Industrial IoT, Industry 4.0 & Predictive Maintenance",
        "task_family": "classification",
        "recommended_model": "random_forest_classifier",
        "comparator_model": "gradient_boosting_classifier",
        "primary_objective": "Predicting rare industrial CNC spindle bearing failures (4.2% failure incidence) from high-frequency multi-sensor telemetry to schedule proactive maintenance before catastrophic assembly line stoppage.",
        "target_variable": "Equipment_Failure (Binary: 1 = Component Failure / Breakdown, 0 = Operating Normally)",
        "imbalance_note": "Severe 4.2% failure incidence. High overall accuracy is deceptive; the plant maintenance team prioritizes PR-AUC and Recall to prevent £45,000/hr unplanned assembly line downtime.",
        "clinical_features": [
            "Vibration_mm_s", "Operating_Temp_C", "Rotational_Speed_RPM", "Torque_Nm",
            "Tool_Wear_Hours", "Acoustic_Noise_dB", "Coolant_Pressure_Bar", "Power_Consumption_kW"
        ],
        "description": "Automated industrial IoT condition monitoring pipeline ingesting multi-axial vibration, torque, and thermal sensors to prevent unpredicted catastrophic mechanical failure.",
        "clinical_context": "Synthetic industrial telemetry dataset simulating automated CNC precision machinery and mechanical degradation curves.",
        "takeaway": "In industrial predictive maintenance, component breakdowns are rare (4.2%). An uncalibrated model predicting zero failures achieves 95.8% accuracy but causes catastrophic factory downtime (£45,000/hr). Tree ensembles trained with balanced sub-sampling identify multi-sensor resonance patterns, increasing Recall to 88% while keeping false maintenance dispatches minimal.",
        "is_synthetic": True,
        "disclaimer": "100% synthetic artificial dataset generated purely for educational and machine learning demonstration. Fictional enterprise organization."
    },

    "marketing_conversion_scoring": {
        "scenario_key": "marketing_conversion_scoring",
        "name": "OmniChannel Retail Group: Customer Purchase Propensity & Campaign Response Scoring",
        "organisation_name": "OmniChannel Direct Marketing & Retail Group",
        "fictional_trust": "OmniChannel Direct Marketing & Retail Group",
        "organisation": "OmniChannel Retail Group (Customer Intelligence & Retention Unit)",
        "domain": "Omnichannel Marketing, Propensity Modeling & Campaign Attribution",
        "task_family": "fairness_audit",
        "recommended_model": "algorithmic_fairness_audit",
        "comparator_model": "penalized_logistic_regression",
        "primary_objective": "Predicting customer campaign purchase conversion probability while auditing disparate promotional allocation across demographic segments to eliminate digital redlining.",
        "target_variable": "Purchased_Conversion (Binary: 1 = Purchased within 7 Days, 0 = Did Not Buy)",
        "protected_attribute": "Customer_Demographic_Tier (Urban Core vs Suburban Families vs Regional Rural)",
        "clinical_features": [
            "Session_Dwell_Mins", "Past_Purchases_12m", "Cart_Abandons_Count", "Email_Open_Rate_Pct",
            "Discount_Sensitivity_Index", "Days_Since_Last_Visit", "Mobile_App_User", "Customer_Lifetime_Value_GBP"
        ],
        "description": "Real-time customer conversion propensity engine targeting high-intent e-commerce shoppers with personalized discount incentives, audited to prevent demographic exclusion.",
        "clinical_context": "Synthetic omnichannel marketing dataset evaluating web engagement trajectories and promotional incentive allocations.",
        "takeaway": "In marketing campaign optimization, conversion rates typically average 7.5%. Targeting solely on raw propensity can inadvertently starve price-sensitive demographic tiers of promotions. Auditing Disparate Impact Ratio (DIR >= 0.80) ensures promotional coupons and credit terms are distributed fairly across all customer segments.",
        "is_synthetic": True,
        "disclaimer": "100% synthetic artificial dataset generated purely for educational demonstration. Fictional commercial enterprise."
    },

    "subscription_churn_forecasting": {
        "scenario_key": "subscription_churn_forecasting",
        "name": "PulseStream Digital Subscriptions: Monthly Subscriber Churn & Retention Analytics",
        "organisation_name": "PulseStream Digital Subscriptions & SaaS Platform",
        "fictional_trust": "PulseStream Digital Subscriptions & SaaS Platform",
        "organisation": "PulseStream Media (Subscription Analytics & Customer Success Directorate)",
        "domain": "Digital Subscription Platforms, SaaS Metrics & Customer Lifetime Value",
        "task_family": "classification",
        "recommended_model": "gradient_boosting_classifier",
        "comparator_model": "random_forest_classifier",
        "primary_objective": "Predicting 30-day subscriber cancellation / churn probability to trigger automated customer success workflows before billing renewal.",
        "target_variable": "Subscriber_Churn (Binary: 1 = Cancelled Subscription, 0 = Retained Active)",
        "imbalance_note": "Moderate 11.8% monthly churn rate. Model threshold must be tuned to maximize retention ROI: intervention cost (£15 concierge outreach) vs customer lifetime value (£360/year).",
        "clinical_features": [
            "Weekly_Stream_Hours", "Playback_Buffering_Events", "Days_Since_Last_Login",
            "Support_Tickets_90d", "Tenure_Months", "Plan_Tier_Price", "Payment_Failure_Retry", "Multi_Device_Usage"
        ],
        "description": "Predictive customer lifecycle engine analyzing digital product telemetry to detect early disengagement indicators 30 days prior to contract renewal.",
        "clinical_context": "Synthetic SaaS product telemetry dataset tracking user engagement telemetry and billing renewal events.",
        "takeaway": "Subscriber churn in recurring digital services (11.8% monthly) erodes Monthly Recurring Revenue (MRR). Operating at default p*=0.50 misses 60% of at-risk users. Tuning the threshold to p*=0.26 optimizes F1 and Precision-Recall tradeoffs, enabling customer success teams to intercept 84% of churners with high-ROI retention interventions.",
        "is_synthetic": True,
        "disclaimer": "100% synthetic artificial dataset generated purely for educational demonstration. Fictional digital SaaS subscription platform."
    },

    "supply_chain_demand_forecasting": {
        "scenario_key": "supply_chain_demand_forecasting",
        "name": "MetroLogix Distribution: Regional Warehouse Dispatch Demand & Freight Capacity Forecasting",
        "organisation_name": "MetroLogix Distribution & Global Freight",
        "fictional_trust": "MetroLogix Distribution & Global Freight",
        "organisation": "MetroLogix Distribution (Logistics Informatics & Inventory Directorate)",
        "domain": "Supply Chain Logistics, Freight Fleet Allocation & Warehouse Operations",
        "task_family": "time_series",
        "recommended_model": "holt_winters_exponential_smoothing",
        "comparator_model": "sarima_forecasting",
        "primary_objective": "Forecasting daily regional warehouse pallet dispatch volumes and required freight truck capacity across a 14-day rolling horizon.",
        "target_variable": "Daily_Pallets_Dispatched (Continuous Daily Count: 600 to 1,400 pallets/day)",
        "temporal_structure": "365 daily dispatch records exhibiting a 7-day cyclical cycle (Wednesday mid-week logistics peak, Sunday warehouse lull) and holiday surge dynamics.",
        "clinical_features": [
            "Date", "Day_of_Week", "Is_Weekend", "Black_Friday_Peak_Season", "Previous_Day_Pallets",
            "Rolling_7d_Mean_Dispatches", "Carrier_Fleet_Available"
        ],
        "description": "Enterprise logistics dispatch model calculating daily warehouse loading bay requirements and third-party freight truck bookings 14 days in advance.",
        "clinical_context": "Synthetic logistics time series tracking daily regional distribution center dispatch throughput and transport capacity.",
        "takeaway": "Warehouse dispatches follow a rigid 7-day operational cycle with mid-week retail replenishments peaking 40% higher than weekend deliveries. Triple Exponential Smoothing (Holt-Winters) yields an out-of-sample MAPE of 6.8%, preventing warehouse bottleneck demurrage charges while guaranteeing on-time retailer deliveries.",
        "is_synthetic": True,
        "disclaimer": "100% synthetic artificial dataset generated purely for educational demonstration. Fictional logistics enterprise."
    },

    "custom": {
        "scenario_key": "custom",
        "name": "Custom Guided Machine Learning & Predictive Analytics Playground",
        "organisation_name": "Custom Clinical & Operational Investigation",
        "fictional_trust": "Inner City Teaching Hospital (Analytics Playground)",
        "organisation": "Custom Clinical & Operational Investigation Unit",
        "domain": "Custom Machine Learning & Predictive Modeling Exploration",
        "task_family": "classification",
        "recommended_model": "random_forest_classifier",
        "comparator_model": "penalized_logistic_regression",
        "primary_objective": "User-configured machine learning, operational forecasting, or fairness audit workflow.",
        "target_variable": "User_Target_Endpoint",
        "description": "Interactive laboratory for testing decision boundaries, time series horizons, and algorithmic equity across user-defined parameters.",
        "clinical_context": "Synthetic sandbox playground for exploring algorithmic behavior under custom operational assumptions.",
        "takeaway": "Select your learning paradigm, data structure, and constraints to inspect dynamic decision surfaces, forecast intervals, and equity audits.",
        "is_synthetic": True,
        "disclaimer": "100% synthetic artificial dataset generated purely for educational demonstration."
    }
}

