from dataclasses import dataclass, field

@dataclass
class GeneralConfig:
    # store everything here dynamically
    _config: dict = field(default_factory=dict)

    @classmethod
    def from_args_and_dist(cls, args, dist_info, extra: dict = None):
        args_dict = vars(args).copy()
        dist_dict = dist_info.shared_dict()
        merged = {**args_dict, **dist_dict, **(extra or {})}
        return cls(merged)

    def __getattr__(self, name):
        if name in self._config:
            return self._config[name]
        raise AttributeError(name)

    def as_dict(self):
        return dict(self._config)