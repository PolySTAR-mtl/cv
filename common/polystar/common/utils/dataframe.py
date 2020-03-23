from typing import Any, Iterable, Callable, Union

from pandas import DataFrame


def format_df_column(df: DataFrame, column_name: str, fmt: Union[Callable, str]):
    df[column_name] = df[column_name].map(fmt.format)


def format_df_columns(df: DataFrame, column_names: Iterable[str], fmt: Union[Callable, str]):
    for c in column_names:
        format_df_column(df, c, fmt)


def format_df_row(df: DataFrame, loc: Any, fmt: Union[Callable, str]):
    df.loc[loc] = df.loc[loc].map(_make_formater(fmt))


def format_df_rows(df: DataFrame, locs: Iterable[Any], fmt: Union[Callable, str]):
    for loc in locs:
        format_df_row(df, loc, fmt)


def _make_formater(fmt: Union[Callable, str]) -> Callable:
    if isinstance(fmt, str):
        return fmt.format
    return fmt
