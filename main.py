from inference.predict import predict_digit
from ollama.gemma import explain_prediction


def main():
    # Path to a sample handwritten digit image
    image_path = "data/sample_digits/0.png"

    # Step 1 & 2: Predict digit and confidence
    digit, confidence = predict_digit(image_path)

    # Step 3: Get explanation from Gemma
    explanation = explain_prediction(digit, confidence)

    # Step 4: Print results
    print("Prediction Result")
    print("-----------------")
    print(f"Predicted Digit : {digit}")
    print(f"Confidence      : {confidence:.2f}\n")

    print("Gemma Explanation")
    print("-----------------")
    print(explanation)


if __name__ == "__main__":
    main()
