import urllib.request
import json

url = "http://localhost:8000/predict"
data = {"HighBP":"1","HighChol":"1","BMI":"25","Smoker":"0","Income":"5","Education":"5","HealthScore":"3"}
req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req) as resp:
        print(resp.read().decode())
except Exception as e:
    import urllib.error
    if isinstance(e, urllib.error.HTTPError):
        print(e.read().decode())
    else:
        print(e)
