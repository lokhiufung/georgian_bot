from typing import Union

import pandas as pd


class TokenFilter(object):
    def __init__(self, ref_file: str, header: Union[int, None]=0):
        if ref_file.endswith('.xlsx'):
            df = pd.read_excel(ref_file, header=header, sheet_name=0)
            self.tokens = df.iloc[:, 0].toList()
        elif ref_file.endswith('.csv'):
            df = pd.read_csv(ref_file, header=header)
            self.tokens = df.iloc[:, 0].toList()
        elif ref_file.endswith('.txt'):
            self.tokens = []
            with open(ref_file, 'r') as f:
                if header is not None:
                    f.readline()  # skip the first line
                for line in f.read():
                    token = line.strip()
                    self.tokens.append(token)
        else:
            raise ValueError(f'File type of {ref_file} is not supported')
        
    def filter(self, tokens):
        tokens = [token for token in tokens if token in self.tokens]
        return tokens

            