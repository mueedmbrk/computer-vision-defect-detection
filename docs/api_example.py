import requests

with open("sample.jpg", "rb") as image:
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        files={"file": ("sample.jpg", image, "image/jpeg")},
        timeout=30,
    )

response.raise_for_status()
print(response.json())
