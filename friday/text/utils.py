# import json
# import pandas as pd

# from friday.text.text_pipeline import build_cantonese_tts_text_pipeline


# cantonese_tts_text_pipeline = build_cantonese_tts_text_pipeline()


# def manifest_to_df(manifest_file: str) -> pd.DataFrame:
#     docs = []
#     with open(manifest_file, 'r') as f:
#         for line in f:
#             doc = json.loads(line)
#             docs.append(doc)
#     df = pd.DataFrame(docs)
#     return df


# def csv_to_df(csv_file: str) -> pd.DataFrame:
#     df = pd.read_csv(csv_file, header=None, names=['audio_filepath', 'duration', 'text'])
#     return df


# def write_manifest(df):
#     with open('manifest.json', 'w') as f:
#         for doc in df.to_dict(orient='records'):
#             f.write(json.dumps(doc, ensure_ascii=False) + '\n')


# def extract_tts(df, spk_id):
#     df = df['spk_id' == spk_id]
#     return df


# def transform_tts(df):
#     df['text'] = df['text'].apply(cantonese_tts_text_pipeline)
#     return df
