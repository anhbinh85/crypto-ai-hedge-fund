def format_price(price):
    if price >= 1:
        return f"${price:.2f}"
    elif price >= 0.01:
        return f"${price:.4f}"
    else:
        return f"${price:.6f}"

# Use format_price for all price displays
# ... find all price displays in consult_crypto and replace with format_price ... 