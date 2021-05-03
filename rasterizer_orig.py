import numpy as np
import freetype
import time
import operator

from scipy.ndimage import label
from fontParts.world import OpenFont, NewFont
from fontParts.fontshell.font import RFont
from fontParts.fontshell.glyph import RGlyph
from pathlib import Path
from fontTools.ttLib.ttFont import TTFont
from fractions import Fraction
from math import hypot, atan2, tan
from typing import Iterator, List, Tuple, Dict


time_counter = {}


def timer(func):
    def new_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_diff = end_time - start_time
        if func.__qualname__ not in time_counter:
            time_counter[func.__qualname__] = 0
        time_counter[func.__qualname__] += time_diff
        return result
    return new_func


def bits(x):
    data = []
    for i in range(8):
        data.insert(0, int((x & 1) == 1))
        x = x >> 1
    return data


def get_aglfn():
    data = {}
    with open(base / "aglfn.txt", 'r') as input_file:
        for line in input_file.read().splitlines():
            if not line.startswith("#"):
                unicode_, glyphname, _ = line.split(";")
                data[glyphname] = int(unicode_, 16)
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
    numPoints = len(coordinates)
    offsetPoints = []
    for i in range(numPoints):
        x, y = coordinates[i % len(coordinates)]
        if i == 0:
            prev_pt = coordinates[-1]
            d1x, d1y = x - prev_pt[0], y - prev_pt[1]
            vectorLength1 = hypot(d1x, d1y)
        else:
            prev_pt = coordinates[i-1]
            d1x = x - prev_pt[0]
            d1y = y - prev_pt[1]
            vectorLength1 = hypot(d1x, d1y)
        if i+1 == numPoints:
            next_pt = coordinates[0]
        else:
            next_pt = coordinates[i+1]
        d2x = next_pt[0] - x
        d2y = next_pt[1] - y
        if prev_pt is not None:
            dx, dy = offset*d1y/vectorLength1, -offset*d1x/vectorLength1
            if next_pt is not None:
                angle1 = atan2(d1y, d1x)
                angle2 = atan2(d2y, d2x)
                angleDiff = (angle2 - angle1)
                t = offset * tan(angleDiff / 2)
                vx, vy = t * d1x/vectorLength1, t * d1y/vectorLength1
                dx += vx
                dy += vy
        else:
            vectorLength2 = hypot(d2x, d2y)
            dx, dy = offset*d2y/vectorLength2, -offset*d2x/vectorLength2
        offsetPoints.append((x + dx + abs(offset), y + dy + abs(offset)))
    return offsetPoints


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


class BaseHintedGlyph:
    def __init__(self, font: freetype.Face, glyph_name: str, scale_ratio: Fraction) -> None:
        self.font = font
        self.scale_ratio = scale_ratio
        self.glyph_name = glyph_name
        self.unicode = None
        self.offset_left = self.font.glyph.metrics.horiBearingX * scale_ratio
        self.offset_top = self.font.glyph.metrics.horiBearingY * scale_ratio
        self.height = self.font.glyph.metrics.height * scale_ratio
        self.width = self.font.glyph.metrics.horiAdvance * scale_ratio
        self.bitmap = self.get_bitmap()
        self.double_bitmap = np.kron(self.bitmap, np.ones((2, 2)))
        self.black_shapes = self._get_shapes(self._get_ones(), 1)
        self.white_shapes = self._get_shapes(self._get_zeros(), 0)

    def get_bitmap(self) -> np.array:
        bitmap = self.font.glyph.bitmap
        data = []
        for i in range(bitmap.rows):
            row = []
            for j in range(bitmap.pitch):
                row.extend(bits(bitmap.buffer[i * bitmap.pitch + j]))
            data.extend(row[: bitmap.width])
        ar = np.array(data).reshape(bitmap.rows, bitmap.width)
        return ar

    def __repr__(self) -> str:
        visualisation = [f"{super().__repr__()}, unicode:{self.unicode}", repr_ar(self.bitmap), ""]
        return "\n".join(visualisation)

    def _get_ones(self) -> List:
        labels, numL = label(self.double_bitmap)
        return [(labels == i).nonzero() for i in range(1, numL + 1)]

    def _get_zeros(self) -> List:
        structure = ((1, 1, 1), ) * 3
        ar_inverted = np.copy(self.double_bitmap)
        for i in range(len(ar_inverted)):
            for k in range(len(ar_inverted[i])):
                ar_inverted[i][k] = not ar_inverted[i][k]
        labels, numL = label(ar_inverted, structure=structure)
        zeros = [(labels == i).nonzero() for i in range(1, numL + 1)]
        indexes_to_remove = []
        for i, zero in enumerate(zeros):
            shape_coos = list(zip(*zero))
            for coos in shape_coos:
                if (
                    coos[0] == 0
                    or coos[0] == (ar_inverted.shape[0] - 1)
                    or coos[1] == 0
                    or coos[1] == (ar_inverted.shape[1] - 1)
                ):
                    indexes_to_remove.append(i)
                    break
        indexes_to_remove = sorted(indexes_to_remove, reverse=True)
        for index_to_remove in indexes_to_remove:
            zeros.pop(index_to_remove)
        return zeros

    def _get_shapes(self, fields, match: int) -> List[Shape]:
        shapes = []
        for field in fields:
            coordinates_sorted = sorted(list(zip(*field)), key=operator.itemgetter(1))
            start = list(filter(lambda x: x[1] == coordinates_sorted[0][1], coordinates_sorted))[-1]
            shapes.append(Shape(self.border_walker(start, match=match)))
        return shapes

    def border_walker(self, start: Tuple[int, int], match: int = None) -> List[Tuple[int, int]]:
        line, cell = start
        directions = [(+1, 0), (0, +1), (-1, 0), (0, -1)]
        visited = [start]
        shape = [start]
        walking = True
        while walking:
            for i, (direction_line, direction_cell) in enumerate(directions):
                coos = (line + direction_line, cell + direction_cell)
                if coos == shape[0]:
                    walking = False
                    break
                if (
                    coos[0] < 0
                    or coos[1] < 0
                    or coos[0] > self.double_bitmap.shape[0] - 1
                    or coos[1] > self.double_bitmap.shape[1] - 1
                ):
                    continue
                if self.double_bitmap[coos[0]][coos[1]] == match and coos not in visited:
                    visited.append(coos)
                    shape.append(coos)
                    directions = directions[i - 1:] + directions[:i - 1]
                    line, cell = coos
                    break
        return shape


class CurrentHintedGlyph(BaseHintedGlyph):
    def __init__(self, *args, **kwargs):
        self.pixel_size = kwargs.pop('pixel_size')
        super().__init__(*args, **kwargs)

    def _draw_shapes(self, glyph: RGlyph, shapes, offset: int, reverse=False):
        for shape in shapes:
            contour = glyph.contourClass()
            shape_coordinates = [(y*abs(offset), x*abs(offset)) for y, x in shape]
            if reverse:
                shape_coordinates = shape_coordinates[::-1]
            shape_coordinates_offsetted = get_offsets(shape_coordinates, offset=offset/2)
            for y, x in shape_coordinates_offsetted:
                contour.appendPoint((int(round(self.offset_left+x)), int(round(self.offset_top-y))))
            glyph.appendContour(contour)

    def draw(self, font: RFont) -> None:
        glyph = font.newGlyph(self.glyph_name)
        glyph.width = round(self.width)
        glyph.unicode = unicode_dict.get(self.glyph_name, None)
        self._draw_shapes(glyph, self.black_shapes, self.pixel_size/2)
        self._draw_shapes(glyph, self.white_shapes, -self.pixel_size/2, reverse=True)


class FontRasterizer:
    def __init__(self, source_ufo: RFont, hinted_font: freetype.Face, font_size: int) -> None:
        self.source_ufo = source_ufo
        self.hinted_font = hinted_font
        self.font_size = font_size
        self.hinted_font.set_pixel_sizes(0, self.font_size)
        glyph_x = self._get_x_data()
        self.scale_ratio = Fraction(500, glyph_x.height)
        self.pixel_size: Fraction = Fraction(500, len(glyph_x.bitmap))
        self.glyphs: List[CurrentHintedGlyph] = []
        self.hinted_kerning = self.get_hinted_kerning()
        return None

    def get_hinted_kerning(self) -> Dict[Tuple[str, str], int]:
        new_kerning = {}
        for left, right in self.source_ufo.kerning.keys():
            left_representant = self.source_ufo.groups.get(left, (left,))[0]
            right_representant = self.source_ufo.groups.get(right, (right,))[0]
            if (self.source_ufo[left_representant].unicode and self.source_ufo[right_representant].unicode):
                value = self.hinted_font.get_kerning(
                    chr(self.source_ufo[left_representant].unicode),
                    chr(self.source_ufo[right_representant].unicode)
                    )
            new_kerning[(left, right)] = round(value.x * self.scale_ratio)
        return new_kerning

    def __iter__(self) -> Iterator:
        return iter(self.glyphs)

    def _get_x_data(self):
        self.hinted_font.load_glyph(glyph_names.index("x"), freetype.FT_LOAD_TARGET_MONO | freetype.FT_LOAD_RENDER)
        return BaseHintedGlyph(self.hinted_font, "x", 1)

    def _get_hinted_glyph(self, glyph_name: str) -> CurrentHintedGlyph:
        self.hinted_font.load_glyph(glyph_names.index(glyph_name), freetype.FT_LOAD_TARGET_MONO | freetype.FT_LOAD_RENDER)
        return CurrentHintedGlyph(self.hinted_font, glyph_name, self.scale_ratio, pixel_size=self.pixel_size)

    def append_glyph(self, glyph_name: str) -> None:
        glyph = self._get_hinted_glyph(glyph_name)
        glyph.pixel_size = self.pixel_size
        self.glyphs.append(glyph)


if __name__ == "__main__":

    debug = False
    if debug:
        import inspect
        for class_ in [FontRasterizer, Shape, CurrentHintedGlyph]:
            functions = inspect.getmembers(class_, predicate=inspect.isfunction)
            for name, function in functions:
                setattr(class_, name, timer(function))
    base = Path(__file__).parent
    source_ufo = OpenFont(base.parent / "source" / "drawing" / "Regular" / "cubic" / "font.ufo")
    hinted_font_path = base.parent / "source" / "VTT" / "shared" / "VTT.ttf"
    hinted_font = freetype.Face(str(hinted_font_path))
    glyph_names = TTFont(str(hinted_font_path)).getGlyphOrder()
    unicode_dict = get_aglfn()
    rasterized_font = FontRasterizer(source_ufo, hinted_font, 20)
    for glyph_name in "abcdefghijkklmnopqrstuvwyxz":
        rasterized_font.append_glyph(glyph_name)
    ufo = NewFont()
    for glyph in rasterized_font:
        glyph.draw(ufo)
    ufo.kerning.update(rasterized_font.hinted_kerning)
    ufo.save('class.ufo')
    if debug:
        for key, value in sorted(time_counter.items(), key=lambda x: x[1], reverse=True):
            print(key, round(value, 5))
