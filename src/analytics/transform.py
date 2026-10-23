"""Transformación de features para la tabla analítica de Olist — sin fuga temporal."""

import pandas as pd


def agregar_features_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas derivadas de las fechas de pedido y entrega.

    No muta el DataFrame recibido — retorna una copia con las columnas nuevas.

    Columnas nuevas:
        tiempo_entrega_dias: días entre compra y entrega real (float, NaN si no
            se ha entregado).
        retraso_entrega_dias: días entre la entrega real y la fecha estimada.
            Positivo significa que llegó tarde; negativo, que llegó antes.
        entregado_tarde: True si retraso_entrega_dias > 0.
        dia_semana_compra: día de la semana de la compra (0=lunes, 6=domingo).
        es_fin_de_semana: True si dia_semana_compra es sábado o domingo.
        mes_compra: mes de la compra (1-12).

    Args:
        df: DataFrame con columnas order_purchase_timestamp,
            order_delivered_customer_date, order_estimated_delivery_date
            (todas datetime64).

    Returns:
        Copia de df con las 6 columnas nuevas agregadas.
    """
    resultado = df.copy()
    compra = resultado["order_purchase_timestamp"]
    entrega = resultado["order_delivered_customer_date"]
    estimada = resultado["order_estimated_delivery_date"]

    resultado["tiempo_entrega_dias"] = (entrega - compra).dt.days.astype("float64")
    resultado["retraso_entrega_dias"] = (entrega - estimada).dt.days.astype("float64")
    resultado["entregado_tarde"] = resultado["retraso_entrega_dias"] > 0
    resultado["dia_semana_compra"] = compra.dt.dayofweek
    resultado["es_fin_de_semana"] = resultado["dia_semana_compra"] >= 5
    resultado["mes_compra"] = compra.dt.month
    return resultado


def agregar_encoding_estado(
    df: pd.DataFrame,
    columna: str = "customer_state",
) -> pd.DataFrame:
    """Agrega una columna de frequency encoding para una columna categórica.

    No muta el DataFrame recibido — retorna una copia con la columna nueva.

    Args:
        df: DataFrame que contiene la columna a codificar.
        columna: nombre de la columna categórica a codificar.

    Returns:
        Copia de df con una columna nueva llamada f"{columna}_freq", donde
        cada fila tiene la frecuencia relativa (0.0 a 1.0) de su categoría
        en el DataFrame completo.
    """
    resultado = df.copy()
    frecuencias = resultado[columna].value_counts(normalize=True)
    resultado[f"{columna}_freq"] = resultado[columna].map(frecuencias)
    return resultado


def agregar_ticket_historico(
    df: pd.DataFrame,
    columna_cliente: str = "customer_id",
    columna_valor: str = "precio_total",
    columna_fecha: str = "order_purchase_timestamp",
) -> pd.DataFrame:
    """Agrega el promedio histórico de compra por cliente, sin fuga temporal.

    Para cada fila, calcula el promedio de columna_valor usando SOLO los
    pedidos anteriores del mismo cliente (ordenados por columna_fecha).
    El primer pedido de cada cliente queda con NaN — no tiene historial previo.

    No muta el DataFrame recibido. El DataFrame retornado puede estar en
    distinto orden de filas que el original (se ordena internamente por
    columna_fecha para calcular la agregación correctamente).

    Args:
        df: DataFrame con las columnas columna_cliente, columna_valor y
            columna_fecha.
        columna_cliente: columna de agrupación (identificador del cliente).
        columna_valor: columna numérica sobre la que se calcula el promedio.
        columna_fecha: columna datetime usada para ordenar cronológicamente.

    Returns:
        Copia de df, ordenada por columna_fecha, con una columna nueva
        llamada "ticket_promedio_historico".
    """
    resultado = df.sort_values(columna_fecha).copy()
    resultado["ticket_promedio_historico"] = resultado.groupby(columna_cliente)[
        columna_valor
    ].transform(lambda serie: serie.shift(1).expanding().mean())
    return resultado
