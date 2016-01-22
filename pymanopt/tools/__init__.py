import collections


def make_enum(name, fields):
    return collections.namedtuple(name, fields)(*range(len(fields)))
