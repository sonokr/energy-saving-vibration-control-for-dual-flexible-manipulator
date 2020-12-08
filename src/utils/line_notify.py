from os import environ

import requests
from dotenv import load_dotenv


def send_line_notify(notification_message="おわったよ"):
    load_dotenv(".env")

    line_notify_token = environ.get("LINE_NOTIFY_TOKEN")
    line_notify_api = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {line_notify_token}"}
    data = {"message": f"{notification_message}"}
    requests.post(line_notify_api, headers=headers, data=data)


if __name__ == "__main__":
    send_line_notify("てすとてすと")
