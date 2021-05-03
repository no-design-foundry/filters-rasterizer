from fontParts.world import OpenFont
from pathlib import Path
from fontTools.ttLib.ttFont import TTFont
from fontTools.ttLib import newTable
from fontTools.ttLib.tables._k_e_r_n import KernTable_format_0


def flatten_kerning(source):

    groups = dict(source.groups)
    kerning = dict(source.kerning)

    flat_kerning = {}

    for (left, right), value in kerning.items():
        left = groups.get(left, (left,))
        right = groups.get(right, (right,))
        for l in left:
            for r in right:
                flat_kerning[(l, r)] = value

    return flat_kerning


if __name__ == "__main__":

    base = Path(__file__).parent

    source = OpenFont(base / "source" / "drawing" / "Regular" / "cubic" / "font.ufo")
    font = TTFont(base / "source" / "VTT" / "exported.ttf")

    flat_kerning = flatten_kerning(source)

    subTable = KernTable_format_0()
    subTable.coverage = 1
    subTable.format = 0
    subTable.kernTable = flat_kerning

    kern = newTable("kern")
    kern.version = 0
    kern.kernTables = [subTable]
    font["kern"] = kern

    font.save(base / "source" / "VTT" / "exported.ttf")
