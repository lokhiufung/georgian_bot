from omegaconf import DictConfig

from friday_server.annoy_search import create_annoy_search_server


annoy_search_server_cfg = {
    'annoy_search': {
        'dim': 512,
        'n_trees': 10,
        'seeder_doc_filepath': 'seeder_search.json'
    },
    'server': {
        'name': 'annoy-search-server'
    }
}

app = create_annoy_search_server(annoy_search_server_cfg=DictConfig(annoy_search_server_cfg))

app.run(debug=True)
