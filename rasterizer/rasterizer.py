import freetype

from scipy.ndimage import label
from numpy import zeros, min as np_min, max as np_max
from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fractions import Fraction
from math import hypot, atan2, tan
from typing import Iterator, List, Tuple
from functools import reduce
from operator import add
from ufoLib2.objects.glyph import Glyph
from ufoLib2.objects.font import Font

try:
    from .rasterize_kerning import rasterize_ufo_kerning
except ImportError:
    from rasterize_kerning import rasterize_ufo_kerning


def bits(x):
    data = []
    for i in range(8):
        value = x & 1
        x = x >> 1
        data.insert(0, value)
        data.insert(0, value)
    return data


def repr_ar(ar):
    visualisation = []
    mapping = {1: "#", 0: " "}
    for row in ar:
        row_str = ""
        for cell in row:
            row_str += mapping[cell]
        visualisation.append(row_str)
    return "\n".join(visualisation)


def get_offsets(coordinates, offset):
    num_points = len(coordinates)
    offset_points = [None] * num_points
    for i in range(num_points):
        x, y = coordinates[i]
        if i == 0:
            prev_pt = coordinates[-1]
            d1x, d1y = x - prev_pt[0], y - prev_pt[1]
            vector_length_1 = hypot(d1x, d1y)
        else:
            prev_pt = coordinates[i - 1]
            d1x = x - prev_pt[0]
            d1y = y - prev_pt[1]
            vector_length_1 = hypot(d1x, d1y)
        if i + 1 == num_points:
            next_pt = coordinates[0]
        else:
            next_pt = coordinates[i + 1]
        d2x = next_pt[0] - x
        d2y = next_pt[1] - y
        if prev_pt is not None:
            dx = offset * d1y / vector_length_1
            dy = -offset * d1x / vector_length_1
            if next_pt is not None:
                angle1 = atan2(d1y, d1x)
                angle2 = atan2(d2y, d2x)
                angleDiff = angle2 - angle1
                t = offset * tan(angleDiff / 2)
                vx = t * d1x / vector_length_1
                vy = t * d1y / vector_length_1
                dx += vx
                dy += vy
        else:
            vector_length_2 = hypot(d2x, d2y)
            dx = offset * d2y / vector_length_2
            dy = -offset * d2x / vector_length_2
        offset_points[i] = (round(x + dx + abs(offset)), round(y + dy + abs(offset)))
    return offset_points


class Shape:
    def __init__(self, matrix_coordinates):
        self.matrix_coordinates = matrix_coordinates
        self.clean()

    def clean(self):
        last_coos = self.matrix_coordinates[0]
        matrix_coordinates_cleaned = []
        index = 1
        for coos in self.matrix_coordinates[1:] + [last_coos]:
            if coos[index] != last_coos[index]:
                matrix_coordinates_cleaned.append(last_coos)
            if coos[0] == last_coos[0]:
                index = 0
            if coos[1] == last_coos[1]:
                index = 1
            last_coos = coos
        self.matrix_coordinates = matrix_coordinates_cleaned

    def __iter__(self):
        return iter(self.matrix_coordinates)

    def __repr__(self):
        return f"{super().__repr__()}, length: {len(self.matrix_coordinates)}"

    def __len__(self):
        return len(self.matrix_coordinates)

    def __getitem__(self, index):
        return self.matrix_coordinates[index]


class CurrentHintedGlyph:
    def __init__(
        self, font: freetype.Face, glyph_name: str, scale_ratio: Fraction, pixel_size
    ) -> None:
        self.pixel_size = pixel_size
        self.scale_ratio = scale_ratio
        self.glyph_name = glyph_name
        self.offset_left = int(round(font.glyph.metrics.horiBearingX * scale_ratio))
        self.offset_top = int(round(font.glyph.metrics.horiBearingY * scale_ratio))
        self.height = font.glyph.metrics.height * scale_ratio
        self.width = int(round(font.glyph.metrics.horiAdvance * scale_ratio))
        self.double_bitmap = self._get_bitmap(font)
        self.black_shapes = self._get_shapes(self._get_ones(), 1)
        self.white_shapes = self._get_shapes(self._get_zeros(), 0)

    def _get_bitmap(self, font):
        bitmap = font.glyph.bitmap
        width = bitmap.width * 2
        buffer = bitmap.buffer
        pitch = bitmap.pitch
        ar = zeros(shape=(bitmap.rows * 2, width))
        for index in range(bitmap.rows):
            row = reduce(add, [bits(buffer[index * pitch + j]) for j in range(pitch)])
            ar[index * 2, :] = row[:width]
            ar[index * 2 + 1, :] = row[:width]
        return ar

    def __repr__(self) -> str:
        visualisation = [super().__repr__(), repr_ar(self.double_bitmap), ""]
        return "\n".join(visualisation)

    def _get_ones(self) -> List:
        labels, numL = label(self.double_bitmap)
        return [(labels == i).nonzero() for i in range(1, numL + 1)]

    def _get_zeros(self) -> List:
        structure = ((1, 1, 1),) * 3
        ar_inverted = 1 - self.double_bitmap
        labels, numL = label(ar_inverted, structure=structure)
        fields = [(labels == i).nonzero() for i in range(1, numL + 1)]
        indexes_to_remove = []
        for i, field in enumerate(fields):
            if any(
                (
                    0 in field[0],
                    0 in field[1],
                    (ar_inverted.shape[0] - 1) in field[0],
                    (ar_inverted.shape[1] - 1) in field[1],
                )
            ):
                indexes_to_remove.append(i)
        indexes_to_remove = sorted(indexes_to_remove, reverse=True)
        for index_to_remove in indexes_to_remove:
            fields.pop(index_to_remove)
        return fields

    def _get_shapes(self, fields, match: int) -> List[Shape]:
        shapes = []
        for field in fields:
            # df = DataFrame({"cell":field[0], "row":field[1]}, columns=["row", "cell"])
            # cell = df.iloc[df[df["row"] == min(df["row"])]["cell"].idxmax()]
            row_min = np_min(field[1])
            cell_max = np_max(field[0][field[1] == row_min])
            shapes.append(Shape(self.border_walker((cell_max, row_min), match=match)))
        return shapes

    def border_walker(
        self, start: Tuple[int, int], match: int = None
    ) -> List[Tuple[int, int]]:
        # This walker starts on the leftmost cell of the lowest row of given. Once it walks around the shape, it returns coordinates of the shape.
        # It requires the bitmap to be doubled in size. So the walker place to make turns.
        cur_line, cur_cell = start
        directions = ((+1, 0), (0, +1), (-1, 0), (0, -1))
        visited = {start}
        shape = [start]
        walking = True
        double_bitmap_shape = self.double_bitmap.shape
        while walking:
            for i, (direction_line, direction_cell) in enumerate(directions):
                line = cur_line + direction_line
                cell = cur_cell + direction_cell
                if (line, cell) == start:
                    walking = False
                    break
                if (
                    line < 0
                    or cell < 0
                    or line > double_bitmap_shape[0] - 1
                    or cell > double_bitmap_shape[1] - 1
                ):
                    continue
                if (
                    self.double_bitmap[line][cell] == match
                    and (line, cell) not in visited
                ):
                    visited.add((line, cell))
                    shape.append((line, cell))
                    directions = directions[i - 1 :] + directions[: i - 1]
                    cur_line = line
                    cur_cell = cell
                    break
        return shape

    def _prepare_coordinates(self, shape, offset):
        shape_coordinates = [
            (
                self.offset_left + x * abs(offset) - self.pixel_size / 2,
                self.offset_top - y * abs(offset) - self.pixel_size / 2,
            )
            for y, x in shape
        ]
        return shape_coordinates

    def _draw_shapes(self, pen, shapes, offset: int, reverse=False):
        for shape in shapes:
            if reverse:
                shape = reversed(shape)
            shape_coordinates = [
                (
                    self.offset_left + x * abs(offset) - self.pixel_size / 2,
                    self.offset_top - y * abs(offset) - self.pixel_size / 2,
                )
                for y, x in shape
            ]
            shape_coordinates_offsetted = get_offsets(
                shape_coordinates, offset=offset / 2
            )
            try:
                pen.endPts.append(pen.endPts[-1] + len(shape_coordinates_offsetted))
            except IndexError:
                pen.endPts.append(len(shape_coordinates_offsetted) - 1)
            pen.points.extend(shape_coordinates_offsetted)
            pen.types.extend([1] * len(shape_coordinates_offsetted))

    def _draw_shapes_ufo(self, glyph, shapes, offset: int, reverse):
        for shape in shapes:
            if reverse:
                shape = reversed(shape)
            shape_coordinates = [
                (
                    self.offset_left + x * abs(offset),
                    self.offset_top - y * abs(offset) - self.pixel_size / 2,
                )
                for y, x in shape
            ]
            shape_coordinates_offsetted = get_offsets(
                shape_coordinates, offset=offset / 2
            )
            pen = glyph.getPen()
            for i, (x, y) in enumerate(shape_coordinates_offsetted):
                if i == 0:
                    pen.moveTo((x, y))
                else:
                    pen.lineTo((x, y))
            pen.closePath()

    def draw(self, output) -> None:
        if isinstance(output, TTFont):
            pen = TTGlyphPen([])
            self._draw_shapes(
                pen, self.black_shapes, self.pixel_size / 2, reverse=False
            )
            self._draw_shapes(
                pen, self.white_shapes, -self.pixel_size / 2, reverse=True
            )
            output["glyf"][self.glyph_name] = pen.glyph()
            output["hmtx"][self.glyph_name] = (self.width, self.offset_left)
        elif isinstance(output, Glyph):
            output.width = self.width
            self._draw_shapes_ufo(
                output, self.black_shapes, self.pixel_size / 2, reverse=False
            )
            self._draw_shapes_ufo(
                output, self.white_shapes, -self.pixel_size / 2, reverse=True
            )


class FontRasterizer:
    def __init__(
        self, hinted_font: freetype.Face, glyph_names: List, font_size: int, x_height
    ) -> None:
        self.glyph_names = glyph_names
        self.hinted_font = hinted_font
        self.font_size = font_size
        self.hinted_font.set_pixel_sizes(0, font_size)
        self.x_height = x_height
        self.scale_ratio = 1 / (self.hinted_font.size.y_scale / 65536)
        self.pixel_size = self.scale_ratio * 64
        self.glyphs: List[CurrentHintedGlyph] = []
        return None

    def __iter__(self) -> Iterator:
        return iter(self.glyphs)

    def _get_hinted_glyph(self, glyph_name: str) -> CurrentHintedGlyph:
        index = self.glyph_names.index(glyph_name)
        self.hinted_font.load_glyph(
            index, freetype.FT_LOAD_TARGET_MONO | freetype.FT_LOAD_RENDER
        )
        return CurrentHintedGlyph(
            self.hinted_font, glyph_name, self.scale_ratio, pixel_size=self.pixel_size
        )

    def rasterize_glyph(self, glyph_name: str) -> None:
        glyph = self._get_hinted_glyph(glyph_name)
        glyph.pixel_size = self.pixel_size
        return glyph


def rasterize(ufo, binary_font, glyph_names_to_process=[], resolution=40, tt_font=None):
    # assert (ufo or binary_font) and not (ufo and binary_font) and (not ufo and not binary_font)
    ufo.info.unitsPerEm = ufo.info.unitsPerEm
    binary_font.seek(0)
    hinted_font = freetype.Face(binary_font)
    if not tt_font:
        tt_font = TTFont(binary_font)
    glyph_names = tt_font.getGlyphOrder()
    try:
        x_height = ufo.info.xHeight
    except AttributeError:
        x_height = 500
    rasterized_font = FontRasterizer(
        hinted_font, glyph_names, int(float(resolution)), x_height
    )
    if not glyph_names_to_process:
        glyph_names_to_process = glyph_names

    for glyph_name in glyph_names_to_process:
        glyph = ufo.get(glyph_name)
        if glyph and len(glyph) > 0:
            rasterized_glyph = rasterized_font.rasterize_glyph(glyph_name)
            glyph.clearContours()
            rasterized_glyph.draw(glyph)

    rasterize_ufo_kerning(ufo, rasterized_font.pixel_size)
    return ufo


def main():
    from extractor import extractUFO
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser(description="Rasterize font glyphs and kerning.")
    parser.add_argument("input_file", type=Path, help="Path to the input font file.")
    parser.add_argument("font_size", type=int, help="Font size of the rasterized font.")
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        help="Path to the output file. If not provided, the output will be saved in the same directory as the input file.",
    )
    parser.add_argument(
        "--glyph_names",
        "-g",
        type=str,
        nargs="+",
        help="List of glyph names to process. If not provided, all glyphs will be processed.",
    )
    parser.add_argument(
        "--no-features",
        "-nf",
        action="store_true",
        default=False,
        help="If set, the features will not be extracted.",
    )

    args = parser.parse_args()

    ufo = Font()

    binary_font = open(args.input_file, "rb")
    input_file = args.input_file
    font_size = args.font_size
    glyph_names_to_process = args.glyph_names if args.glyph_names else ufo.glyphOrder
    output_dir = args.output_dir

    extractUFO(input_file, ufo, doFeatures=not args.no_features)

    rasterized_ufo = rasterize(
        ufo,
        binary_font,
        glyph_names_to_process=glyph_names_to_process,
        resolution=font_size,
    )
    output_file_name = f"{input_file.stem}_{font_size}_rasterized.ufo"
    rasterized_ufo.save(
        output_dir / output_file_name
        if output_dir
        else input_file.parent / output_file_name
    )


if __name__ == "__main__":
    main()
