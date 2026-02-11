def calculate_risk_score(fields: dict, raw_text: str):
    score = 0
    reasons = []

    if not fields.get("amount"):
        score += 50
        reasons.append("Amount not detected")

    if not fields.get("transaction_id"):
        score += 50
        reasons.append("Transaction ID missing")

    if not fields.get("payment_status"):
        score += 55
        reasons.append("Payment status unclear")

    if not fields.get("date"):
        score += 15
        reasons.append("Date not detected")

    suspicious_keywords = [
        "edited", "screenshot", "crop", "photoshop",
        "fake", "duplicate", "copy"
    ]

    lowered = raw_text.lower()
    for word in suspicious_keywords:
        if word in lowered:
            score += 20
            reasons.append(f"Suspicious keyword detected: {word}")
            break

    if score <= 20:
        risk_level = "Low Risk"
    elif score <= 50:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "reasons": reasons
    }
