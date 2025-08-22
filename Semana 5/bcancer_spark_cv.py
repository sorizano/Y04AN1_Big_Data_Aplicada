# -*- coding: utf-8 -*-
"""
Breast Cancer (sintético) con Spark ML:
- Sesión Spark segura en Windows (IPv4, tmp sin espacios, mismo Python del venv)
- Genera data sintética (pandas)
- pandas -> Spark DataFrame (schema explícito)
- Pipeline: Assembler -> Scaler -> LogisticRegression
- Cross-Validation (k=5) para regParam y elasticNetParam
- Métricas en test: AUC, Accuracy, Precision, Recall, F1 + matriz de confusión
- Exporta a Excel en la misma carpeta
Requisitos: pyspark, pandas, numpy, openpyxl (y JDK 8/11/17 instalado)
"""

import os, sys, time, socket
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics


# =============================
# 0) Configuración entorno
# =============================
# Usa el MISMO Python del venv para driver y workers
THIS_PY = sys.executable
os.environ["PYSPARK_PYTHON"] = THIS_PY
os.environ["PYSPARK_DRIVER_PYTHON"] = THIS_PY

# Fuerza IPv4/localhost y tmp sin espacios
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
SPARK_TMP = r"C:\spark-tmp" if os.name == "nt" else "/tmp/spark-tmp"
Path(SPARK_TMP).mkdir(parents=True, exist_ok=True)


# =============================
# 1) Crear Sesión Spark
# =============================
spark = (
    SparkSession.builder
    .appName("BreastCancerSyntheticCV")
    # Si quieres usar todos los núcleos, cambia a "local[*]"
    .master("local[1]")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.local.dir", SPARK_TMP)
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .config("spark.python.worker.reuse", "false")   # más estable en Windows
    .config("spark.port.maxRetries", "64")          # reintentos de puertos
    .config("spark.pyspark.python", THIS_PY)
    .config("spark.pyspark.driver.python", THIS_PY)
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.network.timeout", "300s")
    .getOrCreate()
)

print("Spark:", spark.version, "| Host:", socket.gethostname())
print("Driver Python:", sys.executable)
print("Cuenta JVM:", spark.range(10).count())

# (sanidad) Python del worker
def _pyver_in_worker(it):
    import sys as _sys
    yield "Worker Python: " + _sys.executable
print(spark.sparkContext.parallelize([0], 1).mapPartitions(_pyver_in_worker).first())


# =============================
# 2) Data sintética (pandas)
# =============================
def make_synthetic_breast_cancer(n=1200, seed=42):
    rng = np.random.default_rng(seed)

    features = [
        "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
        "smoothness", "compactness", "concavity", "symmetry", "fractal_dim",
        "mean_density", "cell_size_var", "nuclei_clump", "mitoses"
    ]

    # ~40% malignos
    n_mal = int(n * 0.40)
    n_ben = n - n_mal

    # Benignos
    mu_b = [12.5, 16.0, 80.0, 500.0, 0.090, 0.070, 0.040, 0.18, 0.055, 0.85, 1.2, 2.0, 1.0]
    sd_b = [ 1.5,  2.5, 12.0,  90.0, 0.012, 0.020, 0.015, 0.03,  0.006, 0.08, 0.3, 0.7, 0.6]

    # Malignos
    mu_m = [17.0, 22.5, 110.0, 950.0, 0.105, 0.120, 0.090, 0.21, 0.062, 1.10, 2.0, 3.5, 1.8]
    sd_m = [ 2.0,  3.0,  16.0, 150.0, 0.014, 0.030, 0.020, 0.04,  0.007, 0.10, 0.4, 0.9, 0.7]

    def gen_class(n_rows, mu, sd):
        X = np.column_stack([rng.normal(loc=mu[i], scale=sd[i], size=n_rows) for i in range(len(mu))])
        X = np.maximum(X, 0)  # sin negativos
        # Dependencias suaves realistas
        X[:, 2] = np.maximum(X[:, 2], X[:, 0] * 5 + rng.normal(0, 5, size=n_rows))        # perimeter ~ 5*radius
        X[:, 3] = np.maximum(X[:, 3], (X[:, 0] ** 2) * 3 + rng.normal(0, 80, size=n_rows))# area ~ 3*radius^2
        return X

    X_b = gen_class(n_ben, mu_b, sd_b)
    X_m = gen_class(n_mal, mu_m, sd_m)

    y_b = np.zeros((n_ben, 1), dtype=int)
    y_m = np.ones((n_mal, 1), dtype=int)

    X = np.vstack([X_b, X_m])
    y = np.vstack([y_b, y_m]).ravel()

    # Mezclar
    idx = rng.permutation(len(y))
    X = X[idx]; y = y[idx]

    pdf = pd.DataFrame(X, columns=features)
    pdf["label"] = y
    return pdf

pdf = make_synthetic_breast_cancer(n=1500, seed=123)
print("Filas generadas (pandas):", len(pdf))


# =============================
# 3) pandas -> Spark DF (schema explícito)
# =============================
schema = T.StructType([
    T.StructField(c, T.DoubleType(), nullable=False) for c in pdf.columns if c != "label"
]).add(T.StructField("label", T.IntegerType(), nullable=False))

sdf = spark.createDataFrame(pdf, schema=schema)
print("Conteo en Spark:", sdf.count())
sdf.printSchema()


# =============================
# 4) Train/Test split
# =============================
train_sdf, test_sdf = sdf.randomSplit([0.8, 0.2], seed=2025)
print("Train:", train_sdf.count(), "| Test:", test_sdf.count())


# =============================
# 5) Pipeline de ML
# =============================
feature_cols = [c for c in sdf.columns if c != "label"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
lr = LogisticRegression(featuresCol="features", labelCol="label",
                        predictionCol="prediction", probabilityCol="probability")
pipe = Pipeline(stages=[assembler, scaler, lr])


# =============================
# 6) Cross-Validation (k=5)
# =============================
param_grid = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [0.0, 0.01, 0.05, 0.1])
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
    .build()
)
evaluator = BinaryClassificationEvaluator(labelCol="label",
                                          rawPredictionCol="rawPrediction",
                                          metricName="areaUnderROC")

cv = CrossValidator(
    estimator=pipe,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=5,
    parallelism=0,   # 0 = auto; cuando todo esté estable puedes subirlo
    seed=2025
)

cv_model = cv.fit(train_sdf)
best_model = cv_model.bestModel
bm_lr = [st for st in best_model.stages if isinstance(st, LogisticRegression)][0]
print("CV completada. Mejor LR -> regParam:", bm_lr.getRegParam(),
      "| elasticNetParam:", bm_lr.getElasticNetParam())
cv_avg_auc = float(max(cv_model.avgMetrics))
print("AUC promedio (CV):", round(cv_avg_auc, 4))


# =============================
# 7) Evaluación en TEST
# =============================
pred_test = best_model.transform(test_sdf).cache()
auc_test = float(evaluator.evaluate(pred_test))
print("AUC (test):", round(auc_test, 4))

pred_rdd = pred_test.select("prediction", "label").rdd.map(lambda r: (float(r[0]), float(r[1])))
metrics = MulticlassMetrics(pred_rdd)

accuracy     = float(metrics.accuracy)
precision_1  = float(metrics.precision(1.0))
recall_1     = float(metrics.recall(1.0))
f1_1         = float(metrics.fMeasure(1.0))
cm = metrics.confusionMatrix().toArray()
tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])

print("Accuracy:", round(accuracy, 4))
print("Precision(1):", round(precision_1, 4))
print("Recall(1):", round(recall_1, 4))
print("F1(1):", round(f1_1, 4))
print("Confusion [[TN,FP],[FN,TP]]:\n", cm)


# =============================
# 8) Exportar a Excel (misma carpeta)
# =============================
out_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
ts = time.strftime("%Y%m%d_%H%M%S")
out_path = out_dir / f"bcancer_sintetico_resultados_{ts}.xlsx"

pdf_all  = sdf.toPandas()
pdf_pred = pred_test.select(*feature_cols, "label", "prediction").toPandas()

df_cv = pd.DataFrame([{"metric": "AUC_CV_avg", "value": cv_avg_auc}])
df_test_metrics = pd.DataFrame([{
    "AUC_test": auc_test,
    "Accuracy": accuracy,
    "Precision_pos": precision_1,
    "Recall_pos": recall_1,
    "F1_pos": f1_1
}])
df_conf = pd.DataFrame([[tn, fp], [fn, tp]],
                       columns=["Pred_0", "Pred_1"],
                       index=["Real_0", "Real_1"])

with pd.ExcelWriter(out_path, engine="openpyxl") as w:
    pdf_all.to_excel(w, index=False, sheet_name="data_sintetica")
    pdf_pred.to_excel(w, index=False, sheet_name="pred_test")
    df_cv.to_excel(w, index=False, sheet_name="cv_metrics")
    df_test_metrics.to_excel(w, index=False, sheet_name="best_model_metrics")
    df_conf.to_excel(w, sheet_name="confusion_matrix")

print(f"Excel guardado en: {out_path.resolve()}")

print("Proceso completo OK.")
# spark.stop()  # descomenta si quieres cerrar Spark explícitamente
