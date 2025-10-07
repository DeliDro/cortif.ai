
import requests

url = "localhost:8000/api/v1/predict"

payload = '{"data": {"sepal_length":.1, "sepal_width":.1, "petal_length":.1, "petal_width":.1}}'
headers = {
  'Authorization': 'Bearer be549856564e06e4b73cf3a5fb5f14911a3c11972a4119f5f67692455b3b86ae',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)