import os
from datetime import datetime, timedelta, timezone
from pymongo.mongo_client import MongoClient
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

CEX_LIST = [
    'bitfinex', 'indodax', 'htx', 'gemini', 'korbit', 'binance', 'okx',
    'gate.io', 'kucoin', 'coinex', 'bitrue', 'bitkub', 'bitget', 'coinw',
    'bybit', 'deribit', 'xt.com', 'mexc', 'btse', 'pionex', 'hashkey exchange',
    'coindcx', 'bingx', 'crypto.com exchange', 'deepcoin', 'tapbit', 'weex',
    'toobit', 'flipster', 'blofin', 'bitunix', 'bvox', 'orangex',
    'backpack exchange', 'hashkey global', 'ourbit', 'arkham',
    'huobi', 'kraken', 'coinbase', 'robinhood', 'bitstamp', 
    'cashapp', 'bitvavo', 'crypto.com', 'coinbase institutional'
]
CEX_LIST = [c.lower() for c in CEX_LIST]


def get_mongo_client():
    user = quote_plus(os.getenv("MONGODB_USERNAME"))
    pwd = quote_plus(os.getenv("MONGODB_PASSWORD"))
    cluster = os.getenv("MONGODB_CLUSTER")
    uri = f"mongodb+srv://{user}:{pwd}@{cluster}/?retryWrites=true&w=majority&appName=Cluster0"
    return MongoClient(uri)


def extract_number(val):
    if isinstance(val, dict):
        return float(val.get("$numberInt", val.get("$numberDouble", 0)))
    elif isinstance(val, (int, float)):
        return float(val)
    else:
        return 0.0


def get_cex_btc_data(last: str):
    db_name = os.getenv("DATABASE_NAME")
    coll_name = os.getenv("COLLECTION_NAME")
    client = get_mongo_client()
    db = client[db_name]
    coll = db[coll_name]

    now = datetime.now(timezone.utc)
    if last == "latest":
        doc = coll.find_one(sort=[("timestamp", -1)])
        docs = [doc] if doc else []
    else:
        try:
            hours = int(last.replace("hour", "").replace("s", ""))
            since = now - timedelta(hours=hours)
            docs = list(coll.find({"timestamp": {"$gte": since}}))
        except Exception:
            return None, "Invalid 'last' parameter. Use 'latest', '1hour', ..., '24hours'."

    if not docs:
        return None, "No data found for the selected timeframe."

    docs = [d for d in docs if d is not None]

    # Aggregate inflow/outflow per CEX using per-doc price
    inflow_btc = {}
    inflow_usd = {}
    outflow_btc = {}
    outflow_usd = {}
    latest_price = 0
    SATOSHI = 1e8
    for doc in docs:
        price_info = doc.get("priceInfo", None)
        price = 0
        if isinstance(price_info, dict):
            price = price_info.get("usd", 0)
            if price:
                latest_price = price
        for entry in doc.get("cexFlows", []):
            label = entry.get("label", "").lower()
            if label in CEX_LIST:
                inflow_val = extract_number(entry.get("inflow"))
                outflow_val = extract_number(entry.get("outflow"))
                inflow_btc[label] = inflow_btc.get(label, 0) + inflow_val / SATOSHI
                outflow_btc[label] = outflow_btc.get(label, 0) + outflow_val / SATOSHI
                inflow_usd[label] = inflow_usd.get(label, 0) + (inflow_val / SATOSHI) * price
                outflow_usd[label] = outflow_usd.get(label, 0) + (outflow_val / SATOSHI) * price

    result = []
    for cex in sorted(set(list(inflow_btc.keys()) + list(outflow_btc.keys()))):
        result.append((cex, inflow_btc.get(cex, 0), inflow_usd.get(cex, 0), outflow_btc.get(cex, 0), outflow_usd.get(cex, 0)))

    return result, latest_price 