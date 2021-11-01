import os
from datetime import datetime

import scipy.io.wavfile as wav
from telegram import Bot, Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, callbackcontext

from friday.common import Audio
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

        self.audio_dir = config['temp']['audio']

    def _handle_voice(self, update: Update, context: CallbackContext):
        """
        {'channel_chat_created': False, 'new_chat_photo': [], 'delete_chat_photo': False, 'supergroup_chat_created': False, 'message_id': 30, 'entities': [], 'caption_entities': [], 'date': 1635781951, 'chat': {'id': 329319724, 'first_name': 'albertNotEinstein', 'username': 'molamolahehe', 'type': 'private'}, 'voice': {'file_id': 'AwACAgUAAxkBAAMeYYANP7IaiLYNhA2z5RenaoiDnAoAAtUDAALpaQFUjq8xqPa-01whBA', 'mime_type': 'audio/ogg', 'file_unique_id': 'AgAD1QMAAulpAVQ', 'file_size': 4395, 'duration': 1}, 'new_chat_members': [], 'group_chat_created': False, 'photo': [], 'from': {'username': 'molamolahehe', 'first_name': 'albertNotEinstein', 'id': 329319724, 'is_bot': False, 'language_code': 'en'}}
        """
        
        user = update.effective_user
        telegram_file = update.message.voice.get_file()
        # TODO: io of audio file may be too fucking slow
        audio_filepath = os.path.join(self.audio_dir, '{}-{}.oga'.format(user.id, datetime.now().strftime('%Y_%m_%d_%H_%M_%s')))
        telegram_file.download(
            custom_path=audio_filepath
        )
        
        audio = Audio.from_ogg(audio_filepath, sampling_rate=16000)  # default sampling rate of telegram audio file is 16kHz

        
        signal = audio.get_np_array()


        dialog_output, fulfillment = self.agent.act(
            obs={'audio': signal[:, 0]}  # (n_channels, n_samples)
        )

        update.message.reply_text(dialog_output.reply)

    def _handle_message(self, update: Update, context: CallbackContext):
        
        dialog_output, fulfillment = self.agent.act(
            obs={'text': update.message.text},
        )

        update.message.reply_text(dialog_output.reply)


    def _handle_start(self, update: Update, context: CallbackContext):
        user = update.effective_user
        update.message.reply_markdown_v2(
            fr'Hi\! {user.mention_markdown_v2()}',
            reply_markup=ForceReply(selective=True)
        )
    
    def _handle_error(self, update: Update, context: CallbackContext):
        # TODO: better error handling
        print('error caught ~: ', context.error)

    def _handle_help(self, update: Update, context: CallbackContext):
        pass

    def setup(self, mode='debug'):
        
        self.updater.dispatcher.add_handler(
            MessageHandler(Filters.text & ~Filters.command, self._handle_message)
        )
        self.updater.dispatcher.add_handler(
            MessageHandler(Filters.voice, self._handle_voice)
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
