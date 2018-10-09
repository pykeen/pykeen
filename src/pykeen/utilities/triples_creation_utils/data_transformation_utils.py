# -*- coding: utf-8 -*-


def _transform_subj_obj_pred_to_internal_foramt(data_path_in, data_path_out):
    with open(data_path_in, 'r', encoding='utf-8') as f1, open(data_path_out, 'w', encoding='utf-8') as f2:
        lines = f1.readlines()
        f2.write('@Comment@ Subject Predicate Object\n')
        for line in lines:
            parts = line.strip().split('\t')
            subject = parts[0]
            object = parts[1]
            predicate = parts[2]
            assert len(parts) == 3
            parts = [subject, predicate, object]
            f2.write('\t'.join(parts) + '\n')


def transform_freebase_to_interal_format(data_path_in, data_path_out):
    _transform_subj_obj_pred_to_internal_foramt(data_path_in, data_path_out)


def transform_wn_18_to_interal_format(data_path_in, data_path_out):
    _transform_subj_obj_pred_to_internal_foramt(data_path_in, data_path_out)
