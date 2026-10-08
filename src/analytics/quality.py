"""Contratos de calidad de datos (pandera) para las tablas de Olist."""

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

SCHEMA_ORDERS = DataFrameSchema(
    columns={
        "order_id": Column(
            str,
            Check.str_length(32, 32),
            nullable=False,
            unique=True,
        ),
        "order_status": Column(
            str,
            Check.isin(
                [
                    "delivered",
                    "shipped",
                    "canceled",
                    "unavailable",
                    "invoiced",
                    "processing",
                    "created",
                    "approved",
                ]
            ),
        ),
        "order_purchase_timestamp": Column(pa.DateTime, nullable=False),
    },
    coerce=True,
)

SCHEMA_ITEMS = DataFrameSchema(
    columns={
        "order_id": Column(str, nullable=False),
        "order_item_id": Column(int, Check.greater_than_or_equal_to(1)),
        "product_id": Column(str, nullable=False),
        "seller_id": Column(str, nullable=False),
        "price": Column(
            float,
            [Check.greater_than(0), Check.less_than_or_equal_to(7_000)],
            nullable=False,
        ),
        "freight_value": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
    },
    coerce=True,
)


def validar_schema(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    nombre: str,
) -> list[str]:
    """Valida df contra schema y retorna la lista de errores encontrados.

    No lanza excepción: captura los errores de pandera y los retorna como
    strings para que el pipeline pueda continuar y registrar todos los problemas.

    Args:
        df: DataFrame a validar.
        schema: schema pandera contra el cual validar.
        nombre: nombre descriptivo del DataFrame para los mensajes de error.

    Returns:
        Lista de strings con los errores. Lista vacía si el schema pasa.
    """
    errores: list[str] = []
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        for _, fila in exc.failure_cases.iterrows():
            errores.append(
                f"[{nombre}] columna={fila['column']} check={fila['check']} "
                f"valor={fila['failure_case']} indice={fila['index']}"
            )
    except pa.errors.SchemaError as exc:
        errores.append(f"[{nombre}] {exc}")
    return errores


def generar_reporte_calidad(
    df_entrada: pd.DataFrame,
    df_salida: pd.DataFrame,
    errores: list[str],
    nombre: str = "dataset",
) -> dict:
    """Genera un reporte de la operación de calidad con trazabilidad completa.

    Imprime con print() el formato:
        [CALIDAD] nombre: N entrada -> M salida (K descartados, X.XX%)

    Args:
        df_entrada: DataFrame antes del filtrado/validación.
        df_salida: DataFrame después del filtrado/validación.
        errores: lista de strings con los errores detectados.
        nombre: nombre descriptivo del dataset para el log.

    Returns:
        Dict con nombre, filas_entrada, filas_salida, filas_descartadas,
        tasa_rechazo (str "X.XX%") y errores.
    """
    n_entrada = len(df_entrada)
    n_salida = len(df_salida)
    descartadas = n_entrada - n_salida
    tasa = descartadas / n_entrada * 100 if n_entrada else 0.0
    tasa_str = f"{tasa:.2f}%"

    print(
        f"[CALIDAD] {nombre}: {n_entrada} entrada -> {n_salida} salida "
        f"({descartadas} descartados, {tasa_str})"
    )
    return {
        "nombre": nombre,
        "filas_entrada": n_entrada,
        "filas_salida": n_salida,
        "filas_descartadas": descartadas,
        "tasa_rechazo": tasa_str,
        "errores": errores,
    }
