import collections


class DialogHistory(object):
    OUTPUT_TYPE = collections.namedtuple(
        'DialogEntity',
        ['tag', 'text']
    )
    def __init__(self, capacity: int = 10, client_tag: str = 'client', bot_tag: str = 'bot'):
        self.capacity = 10
        self.dialogs = collections.deque(maxlen=self.capacity)
        self.bot_tag = bot_tag
        self.client_tag = client_tag
        self.output_type = self.OUTPUT_TYPE

    def __len__(self):
        return len(self.dialogs)
        
    def add_dialog(self, text_input, is_bot=False):
        tag = self.bot_tag if is_bot else self.client_tag 
        self.dialogs.append(self.output_type(tag, text_input))

    def get_dialog(self, n: int = 1, include_bot: bool = False, include_tag: bool = False) -> list:
        dialogs = []
        for dialog in reversed(self.dialogs):
            if not include_bot and dialog.tag == self.bot_tag:
                continue
            else:
                dialogs.append(dialog)
        return dialogs
