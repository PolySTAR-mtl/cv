from typing import Any, Callable, Iterable, Union

from pandas import DataFrame

Format = Union[str, Callable]


def format_df_column(df: DataFrame, column_name: str, fmt: Format):
    df[column_name] = df[column_name].map(fmt.format)


def format_df_columns(df: DataFrame, column_names: Iterable[str], fmt: Format):
    for c in column_names:
        format_df_column(df, c, fmt)


def format_df_row(df: DataFrame, loc: Any, fmt: Format):
    df.loc[loc] = df.loc[loc].map(make_formater(fmt))


def format_df_rows(df: DataFrame, locs: Iterable[Any], fmt: Format):
    for loc in locs:
        format_df_row(df, loc, fmt)


def make_formater(fmt: Format) -> Callable:
    if isinstance(fmt, str):
        return fmt.format
    return fmt


def add_percentages_to_df(df: DataFrame, axis: int) -> DataFrame:
    return df.applymap(str) + df.div(df.sum(axis=axis), axis=(1 - axis)).applymap(" ({:.1%})".format)
