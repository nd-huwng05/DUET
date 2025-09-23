from types import SimpleNamespace
from omegaconf import OmegaConf


class Config:
    def __init__(self, config_file: str):
        self.config_file = config_file
        data = OmegaConf.load(config_file)

        self._dict = data
        self._ns = self._dict_to_namespace(data)

    def _dict_to_namespace(self, d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: self._dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [self._dict_to_namespace(i) for i in d]
        else:
            return d

    def __getattr__(self, name):
        if hasattr(self._ns, name):
            return getattr(self._ns, name)
        raise AttributeError(f"Config has no attribute {name}")

    def add(self, section:str , key:str, value):
        if section not in self._dict:
            self._dict[section] = {}
        self._dict[section][key] = value

    @property
    def dict(self):
        return self._dict
