from prism_rules import (
    calculate_prism_score,
    classify_product,
    generate_recommendation
)

# Sample Product
performance = 8
relevance = 7
innovation = 9
scalability = 6
monetization = 8

# Calculate Score
score = calculate_prism_score(
    performance,
    relevance,
    innovation,
    scalability,
    monetization
)

# Classification
classification = classify_product(score)

# Recommendation
recommendations = generate_recommendation(
    performance,
    relevance,
    innovation,
    scalability,
    monetization
)

# Output
print("\n===== PRISM ANALYSIS =====")
print(f"PRISM Score: {score}")
print(f"Classification: {classification}")

print("\nRecommendations:")
for rec in recommendations:
    print(f"- {rec}")
    