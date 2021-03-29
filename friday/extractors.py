"""Extract docs from raw files"""
from typing import Union, List
import glob
import os
import json

import tqdm

from friday.text.text_pipeline import TextPipeline


def drdc_extractor(drdc_dir, files: Union[str, List[str]]='all', text_pipeline: TextPipeline=None):
    if isinstance(files, str) and files.lower() == 'all':
        json_files = list(glob.glob(os.path.join(drdc_dir, '*.json')))
    
    processed_docs = []
    for json_file in json_files:
        print(f'processing {json_file}...')
        with open(json_file, 'r') as f:
            docs = json.load(f)
            size = len(docs['data'])
            for doc in tqdm.tqdm(docs['data']):
                for paragraph in doc['paragraphs']:
                    processed_doc = {}
                    processed_doc['meta'] = {
                        'doc_id': paragraph['id'],
                        'name': doc['title'],
                        'qas': paragraph['qas']
                    }
                    processed_doc['text'] = paragraph['context']
                    if text_pipeline is not None:
                        processed_doc['text'] = text_pipeline(processed_doc['text'])
                        processed_doc['meta']['name'] = text_pipeline(processed_doc['meta']['name'])
                        for qa in processed_doc['meta']['qas']:
                            for answer in qa['answers']:
                                answer['text'] = text_pipeline(answer['text'])
                            qa['question'] = text_pipeline(qa['question'])
                    processed_docs.append(processed_doc)
    return processed_docs

