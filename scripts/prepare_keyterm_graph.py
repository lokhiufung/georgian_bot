import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tqdm


g = nx.Graph()


def english_cleaner(text):
    text = text.strip()
    return text


def build_and_save_keyterm_graph():
    df = pd.read_excel('')
    df = df.dropna()

    keyterm_en = []
    for text in tqdm.tqdm(df['key_terms_en']):
        keyterms = [english_cleaner(keyterm) for keyterm in text.split(',') if keyterm]
        for keyterm in keyterms:
            if keyterm not in keyterm_en:
                keyterm_en.append(keyterm_en)
                g.add_node(keyterm)
        for i in range(len(keyterms)):
            for j in range(len(keyterms)):
                if i != j:
                    g.add_edge(keyterms[i], keyterms[j])
    nx.write_yaml(g, 'keyterm_graph.yaml')
