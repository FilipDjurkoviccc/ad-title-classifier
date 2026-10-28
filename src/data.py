"""Učitavanje podataka iz MySQL-a i priprema trening/val/test skupova."""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from textprep import normalize

# Pristupni podaci se ne čuvaju u kodu, već u promenljivama okruženja (vidi .env.example).
DB = dict(host=os.environ.get("DB_HOST", "185.29.103.47"), port=int(os.environ.get("DB_PORT", 3306)),
          user=os.environ.get("DB_USER", "zadatak"), password=os.environ.get("DB_PASSWORD"),
          database=os.environ.get("DB_NAME", "zadatak"))


def download(path="../data/ml_dataset.csv"):
    """Povlači tabelu ml_dataset iz baze (ako već nije snimljena lokalno)."""
    if os.path.exists(path):
        return pd.read_csv(path)
    import pymysql
    if not DB["password"]:
        raise RuntimeError("Postavi DB_PASSWORD (vidi .env.example) ili stavi ml_dataset.csv u data/.")
    con = pymysql.connect(charset="utf8mb4", **DB)
    try:
        df = pd.read_sql("SELECT * FROM ml_dataset", con)
    finally:
        con.close()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def clean(df):
    """Normalizacija, izbacivanje duplikata i naslova koji se pojavljuju u više grupa."""
    df = df.copy()
    df["text"] = df["name"].map(normalize)
    df = df[df["text"] != ""]
    n_groups = df.groupby("text")["group_id"].nunique()
    df = df[~df["text"].isin(n_groups[n_groups > 1].index)]
    return df.drop_duplicates("text").reset_index(drop=True)


def split(df, seed=42):
    """Stratifikovana podela 80 / 10 / 10."""
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=seed)
    return train, val, test
