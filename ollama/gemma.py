import json
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:2b"


def explain_prediction(digit, confidence):
    """
    digit: int (0–9)
    confidence: float (0–1)
    returns: explanation string from Gemma
    """

    prompt = (
        f"A handwritten digit recognition model predicted the digit '{digit}' "
        f"with a confidence of {confidence:.2f}. "
        f"Explain in simple terms what this confidence means and why the model "
        f"might be sure about this prediction. "
        f"Do not guess the image. Do not classify anything."
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"]
