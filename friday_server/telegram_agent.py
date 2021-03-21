"""
backend server for agent running as a Telegram bot 
"""
from flask import Flask, request
import telegram
from telegram.ext import Dispatcher, MessageHandler, CommandHandler, Filters


def create_telegram_bot_agent_server(server_cfg, agent):
    app = Flask(server_cfg.name)

    bot = telegram.Bot(token=server_cfg.telegram.token)

    ########################################
    # dispatcher: handle request from telegram hook
    dispatcher = Dispatcher(bot, None)
    
    def start_handler(bot, update):
        update.message.reply_text(
            server_cfg.welcome_message,
        )
    
    def reply_handler(bot, update):
        text = update.message.text
        user_id = update.message.from_user_id

        response = agent.get_text_response(text, client_id=user_id)
        update.message.reply_text(response.text_answer)

    dispatcher.add_handler(MessageHandler(Filters.text, reply_handler))
    dispatcher.add_handler(CommandHandler('start', start_handler))
    ########################################

    @app.route('/hook', methods=['POST'])
    def webhook_hanlder():
        update = telegram.Update.de_json(request.get_json(), bot)
        dispatcher.process_update(update)
        return 'ok'

    return app