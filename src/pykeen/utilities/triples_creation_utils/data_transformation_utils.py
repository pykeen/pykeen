# -*- coding: utf-8 -*-

"""Reading of Freebase and related data."""

__all__ = [
    'transform_freebase_to_internal_format',
    'transform_wn_18_to_internal_format',
]


def transform_freebase_to_internal_format(data_path_in: str, data_path_out: str) -> None:
    _transform_spo_to_internal_format(data_path_in, data_path_out)


def transform_wn_18_to_internal_format(data_path_in: str, data_path_out: str) -> None:
    _transform_spo_to_internal_format(data_path_in, data_path_out)


def _transform_spo_to_internal_format(data_path_in: str, data_path_out: str) -> None:
    with open(data_path_in, 'r', encoding='utf-8') as file_in, open(data_path_out, 'w', encoding='utf-8') as file_out:
        print('@Comment@ Subject Predicate Object', file=file_out)
        for line in file_in:
            s, p, o = line.strip().split('\t')
            print(s, p, o, sep='\t', file=file_out)
