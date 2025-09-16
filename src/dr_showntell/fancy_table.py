from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple, Union

from rich._loop import loop_first_last, loop_last
from rich._pick import pick_bool
from rich.box import HEAVY, Box
from rich.segment import Segment
from rich.style import Style, StyleType
from rich.table import Table, _Cell
from rich.text import Text


class FancyBox(Box):
    """Extended Box class that supports multiple header rows."""

    def __init__(self, base_box: Box, num_header_rows: int = 1):
        super().__init__(str(base_box))
        self.num_header_rows = num_header_rows
        self.base_box = base_box
        self.heavy_box = HEAVY

    def get_header_separator(
        self,
        fancy_widths: Iterable[Iterable[int]],
        row_index: int,
        last_header_row: bool,
        edge: bool = True,
    ) -> str:
        if fancy_widths[row_index] == fancy_widths[row_index + 1]:
            return self.get_row(
                widths=fancy_widths[row_index],
                level="head-to-row" if last_header_row else "head",
                edge=edge,
            )

        # Filter out zero widths
        fixed_fancy_widths = [[w for w in fwr if w > 0] for fwr in fancy_widths]

        # Make the bottom part of the divider match row below
        row = self.get_row(
            widths=fixed_fancy_widths[row_index],
            level="head",
            edge=edge,
        )

        def fill_breaks(breaks, rind):
            last_w = 0
            for w in fixed_fancy_widths[rind]:
                last_w += w
                if last_w < len(breaks):
                    breaks[last_w] = True
                    last_w += 1

        num_slots = len(row) - 2
        self_break = [False] * num_slots
        below_break = [False] * num_slots
        fill_breaks(self_break, row_index)
        if row_index < len(fancy_widths) - 1:
            fill_breaks(below_break, row_index + 1)

        new_row = [row[0]]
        for i in range(num_slots):
            if self_break[i] and below_break[i]:
                new_row.append(self.heavy_box.head_row_cross)
            elif self_break[i] and not below_break[i]:
                new_row.append(self.heavy_box.foot_row_cross)
            elif not self_break[i] and below_break[i]:
                new_row.append(self.heavy_box.top_divider)
            else:
                new_row.append(self.heavy_box.head_row_horizontal)
        new_row.append(row[-1])
        return "".join(new_row)

    def get_row(
        self,
        widths: Iterable[int],
        level: Literal["head", "head-to-row", "row", "foot", "mid"] = "row",
        edge: bool = True,
    ) -> str:
        if level == "head":
            left = self.heavy_box.head_row_left
            horizontal = self.heavy_box.head_row_horizontal
            cross = self.heavy_box.head_row_cross
            right = self.heavy_box.head_row_right
        elif level == "head-to-row":
            left = self.head_row_left
            horizontal = self.head_row_horizontal
            cross = self.head_row_cross
            right = self.head_row_right
        elif level == "row":
            left = self.row_left
            horizontal = self.row_horizontal
            cross = self.row_cross
            right = self.row_right
        elif level == "mid":
            left = self.mid_left
            horizontal = " "
            cross = self.mid_vertical
            right = self.mid_right
        elif level == "foot":
            left = self.foot_row_left
            horizontal = self.foot_row_horizontal
            cross = self.foot_row_cross
            right = self.foot_row_right
        else:
            raise ValueError(
                "level must be 'head', 'head-to-row', 'row', 'mid' or 'foot'"
            )

        parts: List[str] = []
        append = parts.append
        if edge:
            append(left)
        for last, width in loop_last(widths):
            append(horizontal * width)
            if not last:
                append(cross)
        if edge:
            append(right)
        return "".join(parts)


@dataclass
class HeaderCell:
    content: Union[str, Text]
    span: int = 1  # Number of columns this header spans
    style: Optional[StyleType] = None


class FancyTable(Table):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fancy_headers: List[List[HeaderCell]] = []
        self._fancy_box: Optional[FancyBox] = None

    def add_header_row(self, *headers: Union[str, HeaderCell]) -> None:
        cells = []
        for header in headers:
            if isinstance(header, str):
                cells.append(HeaderCell(content=header))
            elif isinstance(header, HeaderCell):
                cells.append(header)
            else:
                cells.append(HeaderCell(content=str(header)))
        self.fancy_headers.append(cells)
        self._update_fancy_box()

    def _update_fancy_box(self):
        if self.box and len(self.fancy_headers) > 0:
            self._fancy_box = FancyBox(
                self.box, num_header_rows=len(self.fancy_headers)
            )

    def create_spanned_header(
        self, content: Union[str, Text], span: int, style: Optional[StyleType] = None
    ) -> HeaderCell:
        return HeaderCell(content=content, span=span, style=style)

    def _get_header_count(self) -> int:
        return (
            len(self.fancy_headers)
            if self.fancy_headers
            else (1 if self.show_header else 0)
        )

    def _is_header_row(self, index: int) -> bool:
        return index < self._get_header_count()

    def _is_last_header_row(self, index: int) -> bool:
        return index == self._get_header_count() - 1

    def _get_span_info_for_row(self, header_row_index: int) -> List[Tuple[int, int]]:
        if header_row_index >= len(self.fancy_headers):
            return []
        header_row = self.fancy_headers[header_row_index]
        spans = []
        current_col = 0
        for cell in header_row:
            if cell.span > 1:
                spans.append((current_col, cell.span))
            current_col += cell.span
        return spans

    def _is_column_spanned(
        self, header_row_index: int, column_index: int
    ) -> Tuple[bool, int]:
        spans = self._get_span_info_for_row(header_row_index)

        for start_col, span_length in spans:
            if start_col <= column_index < start_col + span_length:
                return True, start_col

        return False, -1

    def _should_skip_divider(
        self, cell_index: int, span_info: List[Tuple[int, int]]
    ) -> bool:
        divider_column = cell_index
        for start_col, span_length in span_info:
            if start_col <= divider_column < start_col + span_length - 1:
                return True
        return False

    def _get_cell_width(
        self,
        cell_index: int,
        default_width: int,
        all_widths: List[int],
        span_info: List[Tuple[int, int]],
    ) -> int:
        for start_col, span_length in span_info:
            if cell_index == start_col:
                span_widths = all_widths[start_col : start_col + span_length]
                dropped_dividers = span_length - 1
                total_width = sum(span_widths) + dropped_dividers
                return total_width
            if cell_index > start_col and cell_index < start_col + span_length:
                return 0
        return default_width

    def _get_cells(self, console, column_index: int, column):
        if not self.fancy_headers:
            yield from super()._get_cells(console, column_index, column)
            return

        collapse_padding = self.collapse_padding
        pad_edge = self.pad_edge
        padding = self.padding
        any_padding = any(padding)

        first_column = column_index == 0
        last_column = column_index == len(self.columns) - 1

        _padding_cache = {}

        def get_padding(first_row: bool, last_row: bool) -> Tuple[int, int, int, int]:
            cached = _padding_cache.get((first_row, last_row))
            if cached:
                return cached
            top, right, bottom, left = padding

            if collapse_padding:
                if not first_column:
                    left = max(0, left - right)
                if not last_row:
                    bottom = max(0, top - bottom)

            if not pad_edge:
                if first_column:
                    left = 0
                if last_column:
                    right = 0
                if first_row:
                    top = 0
                if last_row:
                    bottom = 0
            _padding = (top, right, bottom, left)
            _padding_cache[(first_row, last_row)] = _padding
            return _padding

        raw_cells: List[Tuple[StyleType, str]] = []
        _append = raw_cells.append
        get_style = console.get_style

        if self.show_header and self.fancy_headers:
            for row_index, header_row in enumerate(self.fancy_headers):
                header_content = self._get_header_content_for_column(
                    column_index, header_row
                )
                header_style = get_style(self.header_style or "") + get_style(
                    column.header_style
                )
                _append((header_style, header_content))

        cell_style = get_style(column.style or "")
        for cell in column.cells:
            _append((cell_style, cell))

        if self.show_footer:
            footer_style = get_style(self.footer_style or "") + get_style(
                column.footer_style
            )
            _append((footer_style, column.footer))

        if any_padding:
            from rich.padding import Padding

            _Padding = Padding
            for first, last, (style, renderable) in loop_first_last(raw_cells):
                yield _Cell(
                    style,
                    _Padding(renderable, get_padding(first, last)),
                    getattr(renderable, "vertical", None) or column.vertical,
                )
        else:
            for style, renderable in raw_cells:
                yield _Cell(
                    style,
                    renderable,
                    getattr(renderable, "vertical", None) or column.vertical,
                )

    def _get_header_content_for_column(
        self, column_index: int, header_row: List[HeaderCell]
    ) -> str:
        current_col = 0
        for cell in header_row:
            if current_col <= column_index < current_col + cell.span:
                if cell.span == 1:
                    return str(cell.content)
                else:
                    if column_index == current_col:
                        return str(cell.content)
                    else:
                        return ""
            current_col += cell.span
        return ""

    def _render_row_content_loop(
        self, console, options, widths, row_cell, columns, max_height, row_style
    ):
        cells = []
        get_style = console.get_style
        for cell_index, (width, cell, column) in enumerate(
            zip(widths, row_cell, columns)
        ):
            render_options = options.update(
                width=width,
                justify=column.justify,
                no_wrap=column.no_wrap,
                overflow=column.overflow,
                height=None,
                highlight=column.highlight,
            )
            lines = console.render_lines(
                cell.renderable,
                render_options,
                style=get_style(cell.style) + row_style,
            )
            max_height = max(max_height, len(lines))
            cells.append(lines)
        return cells

    def _get_header_row_separator(self, box, fancy_widths, row_index, show_edge):
        if self._fancy_box and self.fancy_headers:
            return self._fancy_box.get_header_separator(
                fancy_widths=fancy_widths,
                row_index=row_index,
                last_header_row=self._is_last_header_row(row_index),
                edge=show_edge,
            )
        return box.get_row(fancy_widths[row_index], "head", edge=show_edge)

    def _get_col_dividers(self, box_segments, row_index, last_row):
        if last_row:
            return box_segments[2]
        if self._is_header_row(row_index):
            return box_segments[0]
        return box_segments[1]

    def _render(self, console, options, widths):
        table_style = console.get_style(self.style or "")
        border_style = table_style + console.get_style(self.border_style or "")

        _column_cells = (
            self._get_cells(console, column_index, column)
            for column_index, column in enumerate(self.columns)
        )
        row_cells: List[Tuple[_Cell, ...]] = list(zip(*_column_cells))

        fancy_widths = []
        fancy_spans = []
        for row_i in range(len(row_cells)):
            span_info = self._get_span_info_for_row(row_i)
            fancy_spans.append(span_info)
            if len(span_info) == 0:
                fancy_widths.append(widths)
                continue

            row_widths = []
            for col_i in range(len(row_cells[row_i])):
                width = self._get_cell_width(col_i, widths[col_i], widths, span_info)
                row_widths.append(width)
            fancy_widths.append(row_widths)

        if self._fancy_box and self.fancy_headers:
            _box = self._fancy_box.substitute(
                options, safe=pick_bool(self.safe_box, console.safe_box)
            )
        else:
            _box = (
                self.box.substitute(
                    options, safe=pick_bool(self.safe_box, console.safe_box)
                )
                if self.box
                else None
            )
        _box = _box.get_plain_headed_box() if _box and not self.show_header else _box

        new_line = Segment.line()
        columns = self.columns
        show_header = self.show_header
        show_footer = self.show_footer
        show_edge = self.show_edge
        show_lines = self.show_lines
        leading = self.leading

        _Segment = Segment
        if _box:
            box_segments = [
                (
                    _Segment(_box.head_left, border_style),
                    _Segment(_box.head_right, border_style),
                    _Segment(_box.head_vertical, border_style),
                ),
                (
                    _Segment(_box.mid_left, border_style),
                    _Segment(_box.mid_right, border_style),
                    _Segment(_box.mid_vertical, border_style),
                ),
                (
                    _Segment(_box.foot_left, border_style),
                    _Segment(_box.foot_right, border_style),
                    _Segment(_box.foot_vertical, border_style),
                ),
            ]
            if show_edge:
                separator = _box.get_top([w for w in fancy_widths[0] if w > 0])
                yield _Segment(separator, border_style)
                yield new_line
        else:
            box_segments = []

        get_row_style = self.get_row_style
        get_style = console.get_style

        for row_index, (first_row, last_row, row_cell) in enumerate(
            loop_first_last(row_cells)
        ):
            is_header_row = self._is_header_row(row_index) and show_header
            is_footer_row = last_row and show_footer

            basic_row_data = (
                self.rows[row_index - self._get_header_count()]
                if (
                    not is_header_row
                    and not is_footer_row
                    and row_index >= self._get_header_count()
                )
                else None
            )

            max_height = 1
            if is_header_row or is_footer_row:
                row_style = Style.null()
            else:
                row_style = get_style(
                    get_row_style(
                        console,
                        row_index - self._get_header_count()
                        if show_header
                        else row_index,
                    )
                )

            cells = self._render_row_content_loop(
                console=console,
                options=options,
                widths=fancy_widths[row_index],
                row_cell=row_cell,
                columns=columns,
                max_height=max_height,
                row_style=row_style,
            )
            row_height = max(len(cell) for cell in cells)

            def align_cell(cell, vertical, width, style):
                if is_header_row:
                    vertical = "bottom"
                elif is_footer_row:
                    vertical = "top"
                if vertical == "top":
                    return _Segment.align_top(cell, width, row_height, style)
                elif vertical == "middle":
                    return _Segment.align_middle(cell, width, row_height, style)
                return _Segment.align_bottom(cell, width, row_height, style)

            cells[:] = [
                _Segment.set_shape(
                    align_cell(
                        cell,
                        _cell.vertical,
                        width,
                        get_style(_cell.style) + row_style,
                    ),
                    width,
                    max_height,
                )
                for width, _cell, cell, column in zip(
                    fancy_widths[row_index], row_cell, cells, columns
                )
            ]

            if _box:
                if last_row and show_footer:
                    yield _Segment(
                        _box.get_row(widths, "foot", edge=show_edge),
                        border_style,
                    )
                    yield new_line

                left, right, _divider = self._get_col_dividers(
                    box_segments=box_segments, row_index=row_index, last_row=last_row
                )

                divider = (
                    _divider
                    if _divider.text.strip()
                    else _Segment(
                        _divider.text, row_style.background_style + _divider.style
                    )
                )
                for line_no in range(max_height):
                    if show_edge:
                        yield left
                    for cell_index, (last_cell, rendered_cell) in enumerate(
                        loop_last(cells)
                    ):
                        yield from rendered_cell[line_no]
                        if not last_cell:
                            should_skip_divider = self._should_skip_divider(
                                cell_index, fancy_spans[row_index]
                            )
                            if not should_skip_divider:
                                yield divider
                    if show_edge:
                        yield right
                    yield new_line
            else:
                for line_no in range(max_height):
                    for rendered_cell in cells:
                        yield from rendered_cell[line_no]
                    yield new_line

            if _box and self._is_header_row(row_index) and show_header:
                separator = self._get_header_row_separator(
                    box=_box,
                    fancy_widths=fancy_widths,
                    row_index=row_index,
                    show_edge=show_edge,
                )
                yield _Segment(separator, border_style)
                yield new_line

            end_section = basic_row_data and basic_row_data.end_section
            if _box and (show_lines or leading or end_section):
                if (
                    not last_row
                    and not (show_footer and row_index >= len(row_cells) - 2)
                    and not (show_header and is_header_row)
                ):
                    if leading:
                        yield _Segment(
                            _box.get_row(widths, "mid", edge=show_edge) * leading,
                            border_style,
                        )
                    else:
                        yield _Segment(
                            _box.get_row(widths, "row", edge=show_edge),
                            border_style,
                        )
                    yield new_line

        if _box and show_edge:
            yield _Segment(_box.get_bottom(widths), border_style)
            yield new_line
