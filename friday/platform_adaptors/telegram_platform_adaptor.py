from logging import Filter
import os

from telegram import Bot, Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, callbackcontext

from friday.platform_adaptors.base_platform_adaptor import BasePlatformAdaptor


class TelegramPlatformAdaptor(BasePlatformAdaptor):
    """
    ref: https://github.com/python-telegram-bot/python-telegram-bot/blob/master/examples/echobot.py
    doc: https://python-telegram-bot.readthedocs.io/en/stable/telegram.update.html
    """

    def __init__(self, agent, config):
        super().__init__(agent)

        # for webhook
        self.url = config.get('url', '')
        self.port = config.get('port', '')

        self.bot = Bot(token=config['token'])
        self.updater = Updater(token=config['token'])
    

    def _handle_message(self, update: Update, context: CallbackContext):
        
        dialog_output, fulfillment = self.agent.act(
            obs={'text': update.message.text},
            # fulfillment_key=0
        )
        print(update.message.text)
        print(dialog_output.reply)

        update.message.reply_text(dialog_output.reply)


    def _handle_start(self, update, context):
        user = update.effective_user
        print(user.mention_markdown_v2())
        update.message.reply_markdown_v2(
            fr'Hi\! {user.mention_markdown_v2()}',
            reply_markup=ForceReply(selective=True)
        )
    
    def _handle_error(self, update, context):
        pass

    def _handle_help(self, update, context):
        pass

    def setup(self, mode='debug'):
        
        self.updater.dispatcher.add_handler(
            MessageHandler(Filters.text & ~Filters.command, self._handle_message)
        )
        self.updater.dispatcher.add_handler(
            CommandHandler('start', self._handle_start)
        )
        self.updater.dispatcher.add_handler(
            CommandHandler('help', self._handle_help)
        )
        # handle any bot errors
        self.updater.dispatcher.add_error_handler(self._handle_error)

        if mode == 'production':
            self._setup_webhook()
        elif mode == 'debug':
            self._setup_polling()


    def start_server(self, mode='debug'):
        self.setup(mode=mode)
        print('server started !')
        self.updater.idle()

    def _setup_webhook(self):
        self.updater.start_webhook(
            listen='0.0.0.0',
            port=self.port,
            url_path='',
        )
        # self.updater.bot.setWebhook('')

    def _setup_polling(self):
        self.updater.start_polling()
