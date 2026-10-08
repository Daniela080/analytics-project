"""Módulo de ingesta de datos — carga de fuentes externas hacia el pipeline."""

import pandas as pd


def cargar_csv(ruta: str) -> pd.DataFrame:
    """Carga un archivo CSV y verifica que contenga al menos una fila de datos.

    Args:
        ruta: ruta al archivo CSV a cargar.

    Returns:
        DataFrame con los datos cargados.

    Raises:
        FileNotFoundError: si el archivo no existe en la ruta indicada.
        ValueError: si el archivo existe pero no contiene filas de datos.
    """
    df = pd.read_csv(ruta)

    if df.empty:
        raise ValueError(f"El archivo '{ruta}' no contiene filas de datos.")

    return df


def cargar_olist(data_dir: str) -> dict[str, pd.DataFrame]:
    """Carga las 5 tablas principales de Olist desde data_dir.

    Registra con print() la forma (filas x columnas) de cada tabla al cargarla.
    Las columnas de fecha se cargan como datetime64.

    Args:
        data_dir: ruta al directorio que contiene los archivos CSV de Olist.

    Returns:
        Dict con keys 'orders', 'items', 'customers', 'payments', 'reviews'.

    Raises:
        FileNotFoundError: si alguno de los 5 archivos no existe en data_dir.
    """
    from pathlib import Path

    config = {
        "orders": (
            "olist_orders_dataset.csv",
            [
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        ),
        "items": ("olist_order_items_dataset.csv", None),
        "customers": ("olist_customers_dataset.csv", None),
        "payments": ("olist_order_payments_dataset.csv", None),
        "reviews": (
            "olist_order_reviews_dataset.csv",
            ["review_creation_date", "review_answer_timestamp"],
        ),
    }

    tablas: dict[str, pd.DataFrame] = {}
    for clave, (archivo, fechas) in config.items():
        ruta = Path(data_dir) / archivo
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró {archivo} en {data_dir}")
        df = pd.read_csv(ruta, parse_dates=fechas) if fechas else pd.read_csv(ruta)
        print(f"[EXTRACT] {clave}: {df.shape[0]:,} filas x {df.shape[1]} columnas")
        tablas[clave] = df
    return tablas


def join_verificado(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    on: str | list[str],
    how: str = "left",
    nombre: str = "join",
) -> pd.DataFrame:
    """Realiza un merge y verifica que el resultado no multiplique filas.

    Un join que multiplica filas indica que la tabla derecha tiene duplicados
    en la columna clave: un error silencioso sin esta verificación.

    Args:
        df_left: DataFrame izquierdo (define el número de filas esperado).
        df_right: DataFrame derecho.
        on: columna(s) clave del join.
        how: tipo de join ('left', 'inner', 'outer', 'right').
        nombre: nombre descriptivo para el mensaje de error.

    Returns:
        DataFrame resultante del merge.

    Raises:
        AssertionError: si el resultado tiene más filas que df_left.
    """
    resultado = df_left.merge(df_right, on=on, how=how)
    if len(resultado) > len(df_left):
        raise AssertionError(
            f"Join '{nombre}' multiplicó filas: {len(df_left)} -> {len(resultado)} filas. "
            f"La tabla derecha tiene duplicados en {on}."
        )
    return resultado


def construir_dataset_base(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combina las tablas de Olist en un único DataFrame analítico.

    Estrategia de joins:
        - orders x customers: left join on customer_id
        - + payments_agg: left join on order_id (total_pago, n_cuotas)
        - + items_agg: left join on order_id (n_items, ticket_total)

    Args:
        tablas: dict retornado por cargar_olist().

    Returns:
        DataFrame con una fila por pedido (99,441 filas si los datos son completos).
    """
    pagos_agg = (
        tablas["payments"]
        .groupby("order_id")
        .agg(
            total_pago=("payment_value", "sum"),
            n_cuotas=("payment_installments", "max"),
        )
        .reset_index()
    )
    items_agg = (
        tablas["items"]
        .groupby("order_id")
        .agg(
            n_items=("order_item_id", "count"),
            ticket_total=("price", "sum"),
        )
        .reset_index()
    )

    base = join_verificado(
        tablas["orders"], tablas["customers"], on="customer_id", nombre="orders x customers"
    )
    base = join_verificado(base, pagos_agg, on="order_id", nombre="+ payments_agg")
    base = join_verificado(base, items_agg, on="order_id", nombre="+ items_agg")
    return base
