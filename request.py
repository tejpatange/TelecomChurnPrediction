import requests

# Define the API endpoint
url = "http://127.0.0.1:5000/predict"

# Open and send the CSV file
files = {'file': open("D:/Projects/Telecom_Churn_Prediction/Dataset/churn-bigml-20.csv", 'rb')}
response = requests.post(url, files=files)

# Print the response (predictions)
print(response.json())