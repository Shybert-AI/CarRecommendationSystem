from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')
]


model1 = glob.glob("*.py")
mdfg = [
    basename(i)[:-3] for i in model1 if isfile(i) and basename(i) !="__init__.py"
]
