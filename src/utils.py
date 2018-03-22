#python 3
class dotdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError
