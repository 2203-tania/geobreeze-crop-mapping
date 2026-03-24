import sys
from types import ModuleType

class MockClass:
    """A mock class that behaves like a real class for registration."""
    __name__ = 'MockClass'
    __qualname__ = 'MockClass'
    __module__ = 'mmcv'
    def __init_subclass__(cls, **kwargs):
        pass

class MockModule(ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__version__ = '2.1.0'
        self.__package__ = name
        self.__path__ = []
        self.__spec__ = None
        self.__file__ = f'/mock/{name}.py'

    def __getattr__(self, attr):
        # Return a proper mock class for class-like attributes
        mock = type(attr, (MockClass,), {
            '__name__': attr,
            '__qualname__': attr,
            '__module__': self.__name__,
        })
        setattr(self, attr, mock)
        return mock

submodules = [
    'mmcv', 'mmcv.ops', 'mmcv.image', 'mmcv.video', 'mmcv.utils',
    'mmcv.cnn', 'mmcv.cnn.bricks', 'mmcv.cnn.bricks.drop',
    'mmcv.cnn.bricks.transformer', 'mmcv.cnn.bricks.wrappers',
    'mmcv.cnn.utils', 'mmcv.runner', 'mmcv.parallel',
    'mmcv.fileio', 'mmcv.transforms', 'mmcv.transforms.base',
    'mmcv.transforms.processing', 'mmcv.transforms.loading',
    'mmcv.transforms.wrappers','mmcv.transforms.utils',
]

for name in submodules:
    sys.modules[name] = MockModule(name)

sys.modules['mmcv'].__version__ = '2.1.0'

# Patch mmengine registry to ignore re-registration errors
import mmengine.registry
_orig = mmengine.registry.Registry._register_module
def _safe_reg(self, module=None, module_name=None, force=False, **kwargs):
    try:
        return _orig(self, module=module, module_name=module_name, force=force, **kwargs)
    except KeyError:
        pass
mmengine.registry.Registry._register_module = _safe_reg

# Fix cache_randomness to work as a decorator
import types

def _cache_randomness(func):
    return func

transforms_utils = sys.modules['mmcv.transforms.utils']
transforms_utils.cache_randomness = _cache_randomness
