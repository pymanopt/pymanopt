from typing import Any, Iterable, List, Optional, Tuple


class VoidPrinter:
    """Printer that prints its arguments to the void."""

    def print_header(self):
        pass

    def print_row(self, values: Iterable[Any]):
        pass


def print_list(values: List[str]):
    """Join and print the values given in the ``values`` list."""
    print("".join(values))


class ColumnPrinter(VoidPrinter):
    """Printer that formats values in a column layout.

    Args:
        columns: A list of (column name, format string) tuples.
        placeholder_values: Placeholder values to use when calculating the
            column widths.
        column_padding: Number of spaces to insert between columns.
    """

    def __init__(
        self,
        *,
        columns: List[Tuple[str, str]],
        placeholder_values: Optional[List[Any]] = None,
        column_padding: int = 4,
    ):
        self.column_names, format_strings = zip(*columns)
        self.column_formatters = [
            f"{{value:{format_string}}}" for format_string in format_strings
        ]
        self.column_padding = column_padding

        # Compute the column widths.
        if placeholder_values is None:
            placeholder_values = [0] * len(self.column_names)
        self.column_widths = [
            max(
                len(column),
                len(formatter.format(value=value)),
            )
            + self.column_padding
            for column, formatter, value in zip(
                self.column_names, self.column_formatters, placeholder_values
            )
        ]

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
        """Print ``values`` formatted as a row."""
        column_strings = [
            formatter.format(value=value)
            + " " * (column_width - len(formatter.format(value=value)))
            for formatter, column_width, value in zip(
                self.column_formatters, self.column_widths, values
            )
        ]
        print_list(column_strings)
