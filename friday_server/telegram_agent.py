"""
backend server for agent running as a Telegram bot 
"""
import traceback

from flask import Flask, request
import telegram
from telegram.ext import Dispatcher, MessageHandler, CommandHandler, Filters


def create_telegram_bot_agent_server(server_cfg, agent):
    app = Flask(server_cfg.name)
    print(server_cfg.default_answers.welcome_message)
    bot = telegram.Bot(token=server_cfg.telegram.token)

    ########################################
    # dispatcher: handle request from telegram hook
    dispatcher = Dispatcher(bot, None)
    
    def start_handler(update, callback_context):
        update.message.reply_text(
            server_cfg.default_answers.welcome_message
        )
    
    def reply_handler(update, callback_context):
        text = update.message.text
        user_id = update.message.chat_id

        response = agent.get_text_response(text, client_id=user_id)
        update.message.reply_text(response.text_answer)

    def error_handler(update, callback_context):
        update.message.reply_text('error: {}'.format(callback_context.error))
        
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, reply_handler))  # need filter command requests
    dispatcher.add_handler(CommandHandler('start', start_handler))
    dispatcher.add_error_handler(error_handler)
    ########################################

    @app.route('/hook', methods=['POST'])
    def webhook_hanlder():
        update = telegram.Update.de_json(request.get_json(), bot)
        dispatcher.process_update(update)
        return 'ok'

    return app