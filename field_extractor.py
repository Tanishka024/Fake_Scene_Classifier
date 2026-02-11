import re

def extract_fields(text: str):
    text = text.lower()

    fields = {
        "amount": None,
        "date": None,
        "transaction_id": None,
        "payment_status": None,
        "receiver": None
    }

    amount_match = re.search(
        r'(rs|₹)\s?\d+[.,]?\d*',
        text
    )
    if amount_match:
        fields["amount"] = amount_match.group()

    date_match = re.search(
        r'\b\d{1,2}\s?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s?\d{2,4}?\b',
        text
    )
    if date_match:
        fields["date"] = date_match.group()

    txn_match = re.search(
        r'\b\d{10,16}\b',
        text
    )
    if txn_match:
        fields["transaction_id"] = txn_match.group()

    if any(k in text for k in ["success", "successful", "paid"]):
        fields["payment_status"] = "success"
    elif any(k in text for k in ["failed", "declined"]):
        fields["payment_status"] = "failed"

    receiver_match = re.search(
        r'(to|paid to|merchant)\s+[a-z0-9\s]{3,}',
        text
    )
    if receiver_match:
        fields["receiver"] = receiver_match.group()

    return fields