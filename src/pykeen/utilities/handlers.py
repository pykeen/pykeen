# -*- coding: utf-8 -*-

"""Handler for NDEx."""

import logging

import numpy as np

__all__ = ['handle_ndex']

log = logging.getLogger(__name__)


def handle_ndex(network_uuid: str) -> np.ndarray:
    """Load an NDEx network.

    Example network UUID: f93f402c-86d4-11e7-a10d-0ac135e8bacf
    """
    import ndex2
    ndex_client = ndex2.Ndex2()

    log.info(f'downloading {network_uuid} from ndex')
    res = ndex_client.get_network_as_cx_stream(network_uuid)
    res_json = res.json()

    triples = []

    for entry in res_json:
        for aspect, data in entry.items():
            if aspect == 'edges':
                for edge in data:
                    triples.append([
                        str(edge['s']),
                        edge.get('i', 'interacts'),
                        str(edge['t']),
                    ])

    return np.array(triples)
