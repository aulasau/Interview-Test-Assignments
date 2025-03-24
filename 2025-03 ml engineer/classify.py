import argparse
import pandas as pd
from catboost import CatBoostClassifier

def main():
    parser = argparse.ArgumentParser(description="Classify a string using a CatBoost model.")
    parser.add_argument("model_file", help="Path to the saved CatBoost model.")
    parser.add_argument("input_string", help="String to classify.")
    args = parser.parse_args()

    model = CatBoostClassifier()
    model.load_model(args.model_file)

    input_df = pd.DataFrame({'text': [args.input_string]})
    prediction = model.predict(input_df[['text']])[0]    

    predict_map = {0: 'female', 1: 'male'}
    print(predict_map[prediction])

if __name__ == "__main__":
    main()