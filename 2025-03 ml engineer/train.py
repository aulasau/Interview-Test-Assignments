import argparse
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def read_data(file_path, label, class_name):
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    df = pd.DataFrame({'text': lines, 'label': label, 'class_name': class_name})
    return df


def main():
    parser = argparse.ArgumentParser(description="Train and save a CatBoost model.")
    parser.add_argument("data_folder", help="Path to the folder containing data files.")
    parser.add_argument("model_file", help="Path to save the trained model.")
    args = parser.parse_args()

    female_df = read_data(f"{args.data_folder}/female.txt", 0, 'female')
    male_df = read_data(f"{args.data_folder}/male.txt", 1, 'male')

    # Shuffle the dataset, otherwise catboost does not work properly for some reason :(
    combined_df = pd.concat([female_df, male_df], ignore_index=True).sample(frac=1, random_state=42)


    train_df, valid_df = train_test_split(combined_df, test_size=0.4, random_state=42)
    train_pool = Pool(data=train_df[['text']],
                      label=train_df["label"],
                      text_features=[0],
                      feature_names=['text'])
    valid_pool = Pool(data=valid_df[['text']],
                      label=valid_df["label"],
                      text_features=[0],
                      feature_names=['text'])

    # Step 1: Train a model for evaluation
    eval_model = CatBoostClassifier(
        iterations=300,
        early_stopping_rounds=20,
    )  

    print('Training model with validation dataset to estimate model\'s score on provided data:')
    eval_model.fit(train_pool, eval_set=valid_pool, verbose=50)

    # Evaluate model performance
    valid_preds = eval_model.predict(valid_pool)
    clf_report = classification_report(valid_df["label"], valid_preds)
    print(f"\n\n****************\nClassification report on the validation dataset(40% of the initial data):\n{clf_report}")


    # Step 2: Train final model on the entire dataset
    print(f'\n****************\nTraining final model on the whole dataset!')
    full_pool = Pool(data=combined_df[['text']],
                     label=combined_df["label"],
                     text_features=[0],
                     feature_names=['text'])

    final_model = CatBoostClassifier(
        iterations=300,
        early_stopping_rounds=20,
    )  

    final_model.fit(full_pool, verbose=50)

    # Save the final model
    final_model.save_model(args.model_file)
    print(f"\nFinal model trained on the full dataset and saved to: {args.model_file}")

if __name__ == "__main__":
    main()
