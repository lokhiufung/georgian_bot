def ensure_register_action(method):
    """check whether the agent is registered with an action

    :param method: method
    :type method: [type]
    """
    def wrapper(ref, *args, **kwargs):
        if ref.action is None:
            raise AttributeError('Please register_action() first.')
        return method(ref, *args, **kwargs)
    return wrapper


def preprocess_text(method):
    """use text_pipeline

    :param method: method
    :type method: [type]
    """
    def wrapper(ref, text, *args, **kwargs):
        text = ref.text_pipeline(text)
        return method(ref, *args, **kwargs, text=text)
    return wrapper