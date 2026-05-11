from fuzzywuzzy import process


def classify(messages: list, classifier, allowed_categories: list) -> str:
    outputs = classifier(
        messages,
        max_new_tokens=10,
        do_sample=False,  # Turn off randomness for stability
        pad_token_id=classifier.tokenizer.eos_token_id
    )

    raw_prediction = outputs[0]['generated_text'][-1]['content'].strip()

    # We are looking for the most similar word from allowed_categories
    best_match, score = process.extractOne(raw_prediction.lower(), allowed_categories)

    if score < 50:
        return "unknown"

    return best_match
