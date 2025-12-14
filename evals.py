import json


def eval_retrieval(docs):
    return len(docs) > 0


def eval_reminder_json(output):
    try:
        data = json.loads(output)
        return "schedule" in data
    except Exception:
        return False
