from typing import Any, Iterable, List, Optional, Tuple


class VoidPrinter:
    """Printer that prints nothing."""

    def print_header(self):
        """Print nothing."""

    def print_row(self, values: Iterable[Any]):
        """Print nothing.

        Args:
            values: The values not to print.
        """


def print_list(values: List[str]):
    """Print a formatted list of values.

    Join and print the values given in the ``values`` list.

    Args:
        values: A list of values to join and print.
    """
    print("".join(values))


class ColumnPrinter(VoidPrinter):
    """Printer that formats values in a column layout.

    Args:
        columns: A list of (column name, format string) tuples.
        placeholder_values: Placeholder values to use when calculating the
            column widths.
        column_padding: Number of spaces to insert between columns.

    Attributes:
        column_names: Tuple of column names (headers).
        column_formatters: Tuple of column formatting strings.
        column_padding: The number of spaces used to pad columns.
        column_widths: Tuple of calculated column widths.
    """

    column_names: Tuple[str]
    column_formatters: Tuple[str]
    column_padding: int
    column_widths: Tuple[str]

    def __init__(
        self,
        *,
        columns: List[Tuple[str, str]],
        placeholder_values: Optional[List[Any]] = None,
        column_padding: int = 4,
    ):
        self.column_names, format_strings = map(tuple, zip(*columns))
        self.column_formatters = tuple(
            [f"{{value:{format_string}}}" for format_string in format_strings]
        )
        self.column_padding = column_padding

        # Compute the column widths.
        if placeholder_values is None:
            placeholder_values = [0] * len(self.column_names)
        self.column_widths = tuple(
            [
                max(
                    len(column),
                    len(formatter.format(value=value)),
                )
                + self.column_padding
                for column, formatter, value in zip(
                    self.column_names,
                    self.column_formatters,
                    placeholder_values,
                )
            ]
        )

    def print_header(self):
        """Print a formatted header line."""
        segments = [
            (
                column + " " * (column_width - len(column)),
                "-" * (column_width - self.column_padding)
                + " " * self.column_padding,
            )
            for column, column_width in zip(
                self.column_names, self.column_widths
            )
        ]
        header_segments, underline_segments = zip(*segments)
        print_list(header_segments)
        print_list(underline_segments)

    def print_row(self, values: Iterable[Any]):
        """Print formatted as a row.

        Args:
            values: The values to print.
        """
        column_strings = [
            formatter.format(value=value)
            + " " * (column_width - len(formatter.format(value=value)))
            for formatter, column_width, value in zip(
                self.column_formatters, self.column_widths, values
            )
        ]
        print_list(column_strings)
