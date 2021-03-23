import sys

import requests


def set_webhook(token, webhook_url):
    tg_url = 'https://api.telegram.org/bot{}/setWebhook'.format(token)
    response = requests.get(tg_url, data={'url': webhook_url})
    if response.status_code == 200:
        payload = response.json()
        print(payload)
    else:
        print('failed')


if __name__ == '__main__':
    token = sys.argv[1]
    webhook_url = sys.argv[2]
    set_webhook(
        token=token,
        webhook_url=webhook_url
    )
