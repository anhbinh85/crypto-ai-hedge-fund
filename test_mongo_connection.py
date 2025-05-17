import os
from pymongo.mongo_client import MongoClient
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

def test_connection():
    user = quote_plus(os.getenv("MONGODB_USERNAME"))
    pwd = quote_plus(os.getenv("MONGODB_PASSWORD"))
    cluster = os.getenv("MONGODB_CLUSTER")
    uri = f"mongodb+srv://{user}:{pwd}@{cluster}/?retryWrites=true&w=majority&appName=Cluster0"

    print("MONGODB_USERNAME:", user)
    print("MONGODB_PASSWORD:", pwd)
    print("MONGODB_CLUSTER:", cluster)
    print("MongoDB URI:", uri)

    client = MongoClient(uri)
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print("Connection failed:", e)

if __name__ == "__main__":
    test_connection() 