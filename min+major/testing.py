import os
import numpy as np
import librosa
from keras.models import load_model

def extract_features(audio_path, offset):
    try:
        y, sr = librosa.load(audio_path, offset=offset, duration=3)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
        return mfccs
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

if __name__ == "__main__":
    # Hardcoded paths
    classify_file = '/home/subhash/Documents/major_project/min+major/samples_test/iexample2.wav'
    model_path = '/home/subhash/Documents/major_project/min+major/heartbeat_classifier_normalised.h5'

    if not os.path.isfile(classify_file):
        print(f"File not found: {classify_file}")
        sys.exit(1)

    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    try:
        model = load_model(model_path)
        print(model.summary())
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    x_test = []
    features = extract_features(classify_file, 0.5)
    if features is None:
        sys.exit(1)

    x_test.append(features)
    x_test = np.asarray(x_test)
    print(f"Original x_test shape: {x_test.shape}")

    input_shape = model.input_shape
    print(f"Model input shape: {input_shape}")

    # Reshape x_test according to the model input shape
    if len(input_shape) == 4:
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    elif len(input_shape) == 5:
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2], 1)
    else:
        print(f"Unexpected model input shape: {input_shape}")
        sys.exit(1)

    print(f"Reshaped x_test shape: {x_test.shape}")

    try:
        pred = model.predict(x_test, verbose=1)
        pred_class = np.argmax(pred, axis=1)
        if pred_class[0]:
            print("\nNormal heartbeat")
            print("Confidence:", pred[0][1])
        else:
            print("\nAbnormal heartbeat")
            print("Confidence:", pred[0][0])
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

