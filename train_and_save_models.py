import os
import logging
import joblib
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
#from sklearn.ensemble import AdaBoostClassifier

def train_and_save_models(x_path='X_train_st.csv', y_path='y_train_st.csv', save_dir='models'):
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Load training data
        X_train = pd.read_csv(x_path)
        y_train = pd.read_csv(y_path).squeeze()  # convert DataFrame to Series if needed
        logging.info(f"Loaded X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        return

    # Make sure model directory exists
    os.makedirs(save_dir, exist_ok=True)

    models = {
        'lda_model.pkl': LinearDiscriminantAnalysis(),
        'logReg_model.pkl': LogisticRegression(max_iter=1000),
        'RidgeCl_model.pkl': RidgeClassifier()
    }

    for filename, model in models.items():
        try:
            logging.info(f"Training {filename}...")
            model.fit(X_train, y_train)
            joblib.dump(model, os.path.join(save_dir, filename))
            logging.info(f"Saved model to {os.path.join(save_dir, filename)}")
        except Exception as e:
            logging.error(f"Failed to train/save {filename}: {e}")

if __name__ == "__main__":
    # Default usage: assumes 'X_train.csv' and 'y_train.csv' in the same folder
    train_and_save_models()