def calculate_prism_score(
    performance,
    relevance,
    innovation,
    scalability,
    monetization
):
    """
    Calculate weighted PRISM score.
    """

    score = (
        performance * 0.25 +
        relevance * 0.20 +
        innovation * 0.20 +
        scalability * 0.20 +
        monetization * 0.15
    )

    return round(score, 2)


def classify_product(score):
    """
    Classify product based on PRISM score.
    """

    if score >= 8:
        return "High Potential"

    elif score >= 5:
        return "Moderate Potential"

    else:
        return "Low Potential"


def generate_recommendation(
    performance,
    relevance,
    innovation,
    scalability,
    monetization
):
    """
    Generate recommendation text.
    """

    recommendations = []

    if performance < 5:
        recommendations.append(
            "Improve core product performance."
        )

    if relevance < 5:
        recommendations.append(
            "Increase market relevance and demand alignment."
        )

    if innovation < 5:
        recommendations.append(
            "Focus on innovation and differentiation."
        )

    if scalability < 5:
        recommendations.append(
            "Enhance scalability and growth capability."
        )

    if monetization < 5:
        recommendations.append(
            "Optimize monetization strategy."
        )

    if not recommendations:
        recommendations.append(
            "Product demonstrates strong strategic potential."
        )

    return recommendations
