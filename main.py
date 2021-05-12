import datetime
import time
from fontTools.subset import Subsetter
from fontTools.ttLib import TTFont
import rasterizer
from pathlib import Path

base = Path(__file__).parent

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

start = datetime.datetime.now()
import inspect
for class_ in [rasterizer.FontRasterizer, rasterizer.Shape, rasterizer.CurrentHintedGlyph]:
    functions = inspect.getmembers(class_, predicate=inspect.isfunction)
    for name, function in functions:
        setattr(class_, name, timer(function))
    

tt_font = TTFont(base / ".." / "server" / "test_fonts" / "VTT.ttf")
subsetter = Subsetter()
subsetter.populate(text="no")
subsetter.subset(tt_font)

rasterizer.rasterize(tt_font=tt_font)
tt_font.save("debug.ttf")
end = datetime.datetime.now()
# print((end-start))
# print("="*10)
# for key, value in sorted(time_counter.items(), key=lambda x: x[1], reverse=True):
#     print(key, round(value, 5))