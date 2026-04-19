# Data Ingestion

## Description

Cette branche implémente la phase de **data ingestion** dans Databricks.
Elle consiste à importer les données brutes (CSV) et à les rendre accessibles pour les étapes suivantes du pipeline.

---

## Étapes réalisées

* Upload des fichiers CSV dans un Volume (Unity Catalog)
* Lecture des données avec PySpark
* Vérification basique du schéma

---

## Emplacement des données

```id="f1k9pz"
/Volumes/workspace/default/my_volume/
```

---

## Stockage

Les données sont destinées à être stockées en format Delta Lake pour les étapes suivantes.

---

## Pipeline

```id="p9k2qv"
Ingestion → Cleaning → ML → Visualisation
```

---

## Collaboration

* Données brutes accessibles via Volume
* Utilisées par la branche **cleaning** pour transformation

---
