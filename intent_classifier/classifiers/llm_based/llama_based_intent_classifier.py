import ollama

def classify_intent_using_llama(user_query, model="llama3.1"):
    intents = [
        "convert_to_audio",
        "stock_sentiment",
        "make_notes",
        "get_weather",
        "gmail_operations",
        "analyze_product_sentiment",
        "nl2sql"
    ]

    prompt = f"""
You are an intent classifier.

Classify the user's query into one of the following intents: {intents}

Respond with only the exact intent name. Do not add anything else—no explanations, no formatting.

Examples:
- "Can you read this out loud?" → convert_to_audio
- "What's the trend of Tata Motors stock?" → stock_sentiment
- "Make notes from this image text" → make_notes
- "Tell me the weather in Delhi" → get_weather
- "Show my unread mails" → gmail_operations
- "What do users feel about this new shampoo?" → analyze_product_sentiment
- "Give me 5 rows and any 2 columns of data from the US table" → nl2sql

Query: "{user_query}"
Intent:""".strip()


    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        reply = response['message']['content'].strip()
        predicted_intent = reply.lower().split()[0].strip(":-").strip()
        return predicted_intent
    except Exception as e:
        return f"Error classifying intent: {e}"

if __name__ == "__main__":
    user_query = "Give me 5 rows and any 2 columns of data from the US table"
    intent = classify_intent_using_llama(user_query)
    print(f"Predicted Intent: {intent}")
