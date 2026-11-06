# Databricks notebook source
# MAGIC %md
# MAGIC # Modèle de Churn — Random Forest (scikit-learn)
# MAGIC **Table source : default.churn_model_features**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Chargement de la table et conversion en Pandas

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Chargement depuis Delta Lake
df_spark = spark.table("default.churn_model_features")

# Conversion en Pandas
df = df_spark.toPandas()

print(f"Shape : {df.shape}")
print(f"\nDistribution is_churned :")
print(df["is_churned"].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Définition des colonnes features

# COMMAND ----------



categorical_cols = [
    "Industry",
    "Company_Size",
    "Region",
    "District",
    "Last_Product_1",
    "Last_Product_2",
    "Contract_Status",
    "Payment_Behavior",
    "Campaign_Type",
    "Preferred_Channel",
    "preferred_contact_method",
    "Frequency_of_Purchase",    
]

numerical_cols = [
    "Annual_Revenue_M",
    "Marketing_Spend_K",
    "Days_Since_Last_Purchase",
    "Total_Purchases_Last_Year",
    "Conversion_Rate_Pct",
    "Leads_Generated",
    "nb_employees_contacted",
    "has_decision_maker",
    "nb_decision_makers",
    "influence_score_avg",
    "campaign_response_rate_avg",
    "event_attendance_avg",
    "nb_newsletter_subscribers",
    "tenure_years_avg",
    "active_flag_rate",
    "nb_active_contacts",
]

label_col = "is_churned"

all_cols = categorical_cols + numerical_cols + [label_col]

print(f"Colonnes catégorielles : {len(categorical_cols)}")
print(f"Colonnes numériques    : {len(numerical_cols)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Nettoyage — suppression des lignes avec nulls

# COMMAND ----------

df_clean = df[all_cols].dropna()

print(f"Lignes avant nettoyage : {len(df)}")
print(f"Lignes après nettoyage : {len(df_clean)}")
print(f"Lignes supprimées      : {len(df) - len(df_clean)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Encodage des colonnes catégorielles

# COMMAND ----------

df_model = df_clean.copy()

# LabelEncoder pour chaque colonne catégorielle
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = le

print("Encodage terminé")
print(df_model[categorical_cols].head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Split Train / Test — 80% / 20%

# COMMAND ----------

X = df_model[categorical_cols + numerical_cols]
y = df_model[label_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       
)

print(f"Train : {len(X_train)} lignes")
print(f"Test  : {len(X_test)} lignes")
print(f"\nDistribution train :\n{y_train.value_counts()}")
print(f"\nDistribution test :\n{y_test.value_counts()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Entraînement du modèle Random Forest

# COMMAND ----------

rf = RandomForestClassifier(
    n_estimators=100,   # 100 arbres
    max_depth=5,        # profondeur max de chaque arbre
    random_state=42
)

rf.fit(X_train, y_train)

print("Entraînement terminé")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Prédictions sur le jeu de test

# COMMAND ----------

y_pred      = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:, 1]

# Aperçu des prédictions
results_df = X_test.copy()
results_df["is_churned_reel"]  = y_test.values
results_df["is_churned_predit"] = y_pred
results_df["probabilite_churn"] = y_pred_prob.round(2)

display(spark.createDataFrame(
    results_df[["is_churned_reel", "is_churned_predit", "probabilite_churn"]].head(20)
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Performances du modèle

# COMMAND ----------

print("=" * 45)
print("   PERFORMANCES DU MODÈLE")
print("=" * 45)
print(f"   Accuracy  : {accuracy_score(y_test, y_pred):.4f}  ({accuracy_score(y_test, y_pred)*100:.1f}%)")
print(f"   Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"   Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"   F1-Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"   AUC-ROC   : {roc_auc_score(y_test, y_pred_prob):.4f}")
print("=" * 45)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Matrice de confusion

# COMMAND ----------

cm = confusion_matrix(y_test, y_pred)

print("Matrice de confusion :")
print(f"                  Prédit 0    Prédit 1")
print(f"  Réel 0  (non churné)  {cm[0][0]:>6}      {cm[0][1]:>6}")
print(f"  Réel 1  (churné)      {cm[1][0]:>6}      {cm[1][1]:>6}")

print("\nRapport détaillé :")
print(classification_report(y_test, y_pred, target_names=["Non churné", "Churné"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Feature Importance — colonnes qui influencent le plus le churn

# COMMAND ----------

fi_df = pd.DataFrame({
    "feature":    categorical_cols + numerical_cols,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

fi_df["importance"] = fi_df["importance"].round(4)

print("Top 15 features les plus importantes :")
print(fi_df.head(15).to_string(index=False))

# Visualisation bar chart dans Databricks
display(spark.createDataFrame(fi_df.head(15)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Sauvegarde du modèle avec pickle

# COMMAND ----------

import pickle

# Sérialisation du modèle en bytes
model_bytes    = pickle.dumps(rf)
encoders_bytes = pickle.dumps(encoders)

# Sauvegarde dans Delta Lake
model_df = spark.createDataFrame([
    ("random_forest", bytearray(model_bytes)),
    ("encoders",      bytearray(encoders_bytes))
], ["model_name", "model_data"])

model_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("default.churn_model_registry")

print(" Modèle sauvegardé dans default.churn_model_registry")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pour charger le modèle

# COMMAND ----------

import pickle

registry = spark.table("default.churn_model_registry").toPandas()

rf       = pickle.loads(bytes(registry[registry["model_name"] == "random_forest"]["model_data"].values[0]))
encoders = pickle.loads(bytes(registry[registry["model_name"] == "encoders"]["model_data"].values[0]))

print(" Modèle chargé et prêt")
