from fontTools.ttLib.ttFont import TTFont
from math import ceil


def kerning_indexes(tt_font):
    for feature in tt_font["GPOS"].table.FeatureList.FeatureRecord:
        if feature.FeatureTag == "kern":
            return feature.Feature.LookupListIndex


def round_XAdvance(Value1, pixel_size):
    Value1.XAdvance = round_value(Value1.XAdvance, pixel_size)


def round_value(value, pixel_size):
    return ceil(value / pixel_size) * pixel_size


def update_format_1_subtable(sub_table, pixel_size):
    first_glyphs = sub_table.Coverage.glyphs
    for index, pair in enumerate(sub_table.PairSet):
        for record in pair.PairValueRecord:
            round_XAdvance(record.Value1, pixel_size)


def update_format_2_subtable(sub_table, pixel_size):
    classes_1 = kern_class(sub_table.ClassDef1.classDefs)
    classes_2 = kern_class(sub_table.ClassDef2.classDefs)
    for index_1, class_1 in enumerate(sub_table.Class1Record):
        for index_2, class_2 in enumerate(class_1.Class2Record):
            if index_1 not in classes_1:
                continue
            if index_2 not in classes_2:
                continue
            if class_2.Value1.XAdvance != 0:
                for glyph_1 in classes_1[index_1]:
                    for glyph_2 in classes_2[index_2]:
                        round_XAdvance(class_2.Value1, pixel_size)


def kern_class(class_definition):
    classes = {}
    for glyph, idx in class_definition.items():
        if idx not in classes:
            classes[idx] = []
        classes[idx].append(glyph)
    return classes


def rasterize_kerning(tt_font, pixel_size):
    try:
        indexes = kerning_indexes(tt_font)
        for index in indexes:
            lookup = tt_font["GPOS"].table.LookupList.Lookup[index]
            for sub_table in lookup.SubTable:
                if sub_table.Format == 1:
                    update_format_1_subtable(sub_table, pixel_size)
                elif sub_table.Format == 2:
                    update_format_2_subtable(sub_table, pixel_size)
    except Exception as e:
        print(e)


def rasterize_ufo_kerning(ufo, pixel_size):
    for key, value in [i for i in ufo.kerning.items()]:
        rounded_value = round_value(value, pixel_size)
        if rounded_value != 0:
            ufo.kerning[key] = round(rounded_value)
        else:
            del ufo.kerning[key]
