def validate_product(product, valid_products):
    if product not in valid_products:
        raise ValueError("Invalid product selected")


def validate_question(question):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
