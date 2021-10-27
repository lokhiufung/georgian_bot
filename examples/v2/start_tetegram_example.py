import argparse
import json

from friday.agent_factory import load_example_agent
from friday.platform_adaptors.telegram_platform_adaptor import TelegramPlatformAdaptor



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', type=str, required=True, help='file path of the faq docs in json format')
    parser.add_argument('--config', type=str, required=True, help='file path of the configuration file in json format')

    args = parser.parse_args()

    agent = load_example_agent(docs_filepath=args.docs)

    with open(args.config, 'r') as f:
        config = json.load(f)

    telegram_server = TelegramPlatformAdaptor(agent, config=config)
    
    telegram_server.start_server()
