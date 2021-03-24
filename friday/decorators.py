def ensure_register_action(method):
    def wrapper(ref, *args, **kwargs):
        if ref.action is None:
            raise AttributeError('Please register_action() first.')
        return method(ref, *args, **kwargs)
    return wrapper


def preprocess_text(method):
    def wrapper(ref, text, *args, **kwargs):
        text = ref.text_pipeline(text)
        return method(ref, *args, **kwargs, text=text)
    return wrapper