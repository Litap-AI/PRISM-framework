import pandas as pd
import random

# Number of products
NUM_PRODUCTS = 200

# Product categories
categories = [
    "EV",
    "SaaS",
    "FinTech",
    "Healthcare",
    "EdTech",
    "AI Tools",
    "Cybersecurity",
    "E-commerce"
]

products = []

for i in range(NUM_PRODUCTS):

    # Generate PRISM scores
    performance = random.randint(1, 10)
    relevance = random.randint(1, 10)
    innovation = random.randint(1, 10)
    scalability = random.randint(1, 10)
    monetization = random.randint(1, 10)

    # Weighted PRISM Score
    final_score = (
        performance * 0.25 +
        relevance * 0.20 +
        innovation * 0.20 +
        scalability * 0.20 +
        monetization * 0.15
    )

    # Success Label
    if final_score >= 8:
        success_label = "High"

    elif final_score >= 5:
        success_label = "Medium"

    else:
        success_label = "Low"

    # Product Record
    product = {
        "product_id": i + 1,
        "product_name": f"Product_{i+1}",
        "category": random.choice(categories),
        "performance": performance,
        "relevance": relevance,
        "innovation": innovation,
        "scalability": scalability,
        "monetization": monetization,
        "final_score": round(final_score, 2),
        "success_label": success_label
    }

    products.append(product)

# Create DataFrame
df = pd.DataFrame(products)

# Save CSV
df.to_csv("data/raw/products.csv", index=False)

print("✅ Dataset generated successfully!")
print(df.head())
