from opencc import OpenCC


__t2s = OpenCC('t2s')


def normalize_to_mandarin(text: str) -> str:
    """normalize text to simplified chinese

    :param text: text
    :type text: str
    :return: text in simplified chinese
    :rtype: str
    """
    text = __t2s.convert(text)
    return text



