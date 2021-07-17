import collections
from typing import List 


class SimpleDialogHistory(object):
    OUTPUT_TYPE = collections.namedtuple(
        'DialogEntity',
        ['tag', 'text']
    )
    def __init__(self, capacity: int = 10, client_tag: str = 'client', bot_tag: str = 'bot'):
        """Object that storage the dialog history in memory

        :param capacity: maximum capacity of storage, defaults to 10
        :type capacity: int, optional
        :param client_tag: tag name for client, defaults to 'client'
        :type client_tag: str, optional
        :param bot_tag: tag name for bot, defaults to 'bot'
        :type bot_tag: str, optional
        """
        self.capacity = capacity
        self.dialogs = collections.deque(maxlen=self.capacity)
        self.bot_tag = bot_tag
        self.client_tag = client_tag
        self.output_type = self.OUTPUT_TYPE

    def __len__(self):
        return len(self.dialogs)
        
    def add_dialog(self, text_input: str, is_bot: bool=False):
        """store and append dialog

        :param text_input: text of the dialog
        :type text_input: str
        :param is_bot: whether the text input is from the bot, defaults to False
        :type is_bot: bool, optional
        """
        tag = self.bot_tag if is_bot else self.client_tag 
        self.dialogs.append(self.output_type(tag, text_input))

    def get_dialog(self, n: int = 1, include_bot: bool = False, include_tag: bool = False) -> list:
        """retrieve dialog history

        :param n: number of dialog to traceback, defaults to 1
        :type n: int, optional
        :param include_bot: whehter to include the responses from the bot, defaults to False
        :type include_bot: bool, optional
        :param include_tag: whether to include the tag name (i.e client/bot), defaults to False
        :type include_tag: bool, optional
        :return: list of dialog
        :rtype: List[NamedTuple]
        """
        dialogs = []
        for dialog in reversed(self.dialogs):
            if not include_bot and dialog.tag == self.bot_tag:
                continue
            else:
                dialogs.append(dialog)
        return dialogs
