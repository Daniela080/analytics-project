"""Segmentación de clientes: features RFM, clustering K-Means, validación y perfilado."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def calcular_rfm(pedidos: pd.DataFrame, fecha_referencia: pd.Timestamp) -> pd.DataFrame:
    """Construye la tabla RFM (Recencia, Frecuencia, Monto) por cliente.

    Args:
        pedidos: DataFrame con una fila por pedido y las columnas
            'customer_unique_id', 'order_purchase_timestamp' y 'monto_pedido'.
        fecha_referencia: fecha desde la cual se cuenta la recencia
            (normalmente, un día después del último pedido del dataset).

    Returns:
        DataFrame con una fila por cliente único y las columnas
        'customer_unique_id', 'recencia' (días, int), 'frecuencia'
        (número de pedidos, int) y 'monto' (suma de monto_pedido, float).
    """
    rfm = (
        pedidos.groupby("customer_unique_id")
        .agg(
            ultima_compra=("order_purchase_timestamp", "max"),
            frecuencia=("order_purchase_timestamp", "count"),
            monto=("monto_pedido", "sum"),
        )
        .reset_index()
    )
    rfm["recencia"] = (fecha_referencia - rfm["ultima_compra"]).dt.days.astype(int)
    rfm["frecuencia"] = rfm["frecuencia"].astype(int)
    rfm["monto"] = rfm["monto"].astype(float)
    return rfm[["customer_unique_id", "recencia", "frecuencia", "monto"]]


def entrenar_pipeline_clustering(
    X: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> Pipeline:
    """Construye y entrena un pipeline de clustering K-Means.

    El pipeline tiene dos pasos: escalado estándar y K-Means
    (con n_init=10, para evitar convergencias inestables).

    Args:
        X: features numéricas (ej. recencia, frecuencia, monto).
        n_clusters: número de clusters k.
        random_state: semilla para reproducibilidad.

    Returns:
        Pipeline ya entrenado (fit ya ejecutado), listo para .predict().
    """
    pipeline = Pipeline(
        [
            ("escalador", StandardScaler()),
            ("modelo", KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)),
        ]
    )
    pipeline.fit(X)
    return pipeline


def evaluar_clustering(X, labels) -> dict:
    """Calcula las métricas estándar de validación de un clustering.

    Args:
        X: features usadas para el clustering (mismas dimensiones que se
            usaron para entrenar, sin la columna de segmento).
        labels: etiqueta de cluster asignada a cada fila de X.

    Returns:
        Dict con claves 'silhouette' y 'davies_bouldin' (floats).
    """
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def perfilar_segmentos(
    df: pd.DataFrame,
    columna_segmento: str,
    columnas_perfil: list[str],
) -> pd.DataFrame:
    """Calcula el perfil promedio de cada segmento.

    Args:
        df: DataFrame que incluye la columna de segmento y las columnas
            a perfilar.
        columna_segmento: nombre de la columna con la etiqueta de cluster.
        columnas_perfil: columnas numéricas a promediar por segmento.

    Returns:
        DataFrame indexado por segmento, con el promedio de cada columna
        de columnas_perfil y una columna adicional 'n_clientes' con el
        conteo de filas de cada segmento.
    """
    grupos = df.groupby(columna_segmento)
    perfil = grupos[columnas_perfil].mean()
    perfil["n_clientes"] = grupos.size()
    return perfil
