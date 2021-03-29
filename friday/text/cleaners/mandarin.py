from opencc import OpenCC


__t2s = OpenCC('t2s')


def normalize_to_mandarin(text):
    text = __t2s.convert(text)
    return text



