import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----- 1. EMG Signal Processing Functions -----

def preprocess_emg(data, lowcut=20, highcut=500, fs=1000, notch_freq=50):
    """
    Preprocess EMG data with bandpass and notch filtering, and rectification
    
    Parameters:
    - data: DataFrame with Time[s] and Voltage[V] columns
    - lowcut: Lower cutoff frequency for bandpass filter (Hz)
    - highcut: Upper cutoff frequency for bandpass filter (Hz)
    - fs: Sampling frequency (Hz)
    - notch_freq: Frequency to remove (Hz)
    
    Returns:
    - time: Time values
    - processed_signal: Processed EMG signal
    """
    # Extract time and voltage
    time = data['Time[s]'].values
    emg = data['Voltage[V]'].values
    
    # Bandpass filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, emg)
    
    # Notch filter for power line interference
    b_notch, a_notch = signal.iirnotch(notch_freq, 30, fs)
    notch_filtered = signal.filtfilt(b_notch, a_notch, filtered)
    
    # Full-wave rectification
    rectified = np.abs(notch_filtered)
    
    # Smoothing with moving average filter
    window_size = int(0.05 * fs)  # 50ms window
    smoothed = np.convolve(rectified, np.ones(window_size)/window_size, mode='same')
    
    return time, smoothed

def segment_signal(time, signal, window_size=0.2, overlap=0.1, fs=1000):
    """
    Segment signal into overlapping windows
    
    Parameters:
    - time: Time values
    - signal: EMG signal
    - window_size: Window size in seconds
    - overlap: Overlap between windows in seconds
    - fs: Sampling frequency (Hz)
    
    Returns:
    - segments: List of signal segments
    - segment_times: Start time of each segment
    """
    window_samples = int(window_size * fs)
    overlap_samples = int(overlap * fs)
    step = window_samples - overlap_samples
    
    segments = []
    segment_times = []
    
    for i in range(0, len(signal) - window_samples + 1, step):
        segment = signal[i:i + window_samples]
        segments.append(segment)
        segment_times.append(time[i])
    
    return np.array(segments), np.array(segment_times)

def extract_features(segments):
    """
    Extract time-domain features from EMG segments
    
    Parameters:
    - segments: List of signal segments
    
    Returns:
    - features: Array of features for each segment
    """
    features = []
    
    for segment in segments:
        # Mean Absolute Value (MAV)
        mav = np.mean(np.abs(segment))
        
        # Root Mean Square (RMS)
        rms = np.sqrt(np.mean(np.square(segment)))
        
        # Waveform Length (WL)
        wl = np.sum(np.abs(np.diff(segment)))
        
        # Zero Crossing Rate (ZCR)
        zcr = np.sum(np.abs(np.diff(np.sign(segment)))) / (2 * len(segment))
        
        # Slope Sign Changes (SSC)
        diff = np.diff(segment)
        ssc = np.sum((diff[:-1] * diff[1:]) < 0) / len(segment)
        
        # Integrated EMG (IEMG)
        iemg = np.sum(np.abs(segment))
        
        # Feature vector
        feature_vector = [mav, rms, wl, zcr, ssc, iemg]
        features.append(feature_vector)
    
    return np.array(features)

# ----- 2. Echo State Network Implementation -----

class EchoStateNetwork:
    def __init__(self, n_inputs, n_outputs, n_reservoir=200, spectral_radius=0.95,
                 sparsity=0.1, noise=0.001, input_scaling=1.0, leaking_rate=0.3,
                 random_state=42):
        """
        Initialize Echo State Network
        
        Parameters:
        - n_inputs: Number of input features
        - n_outputs: Number of output classes
        - n_reservoir: Size of the reservoir
        - spectral_radius: Spectral radius of the reservoir
        - sparsity: Proportion of non-zero weights in reservoir
        - noise: Noise added during training
        - input_scaling: Scaling of input weights
        - leaking_rate: Leaking rate for reservoir neurons
        - random_state: Random seed for reproducibility
        """
        # Set random seed
        np.random.seed(random_state)
        
        # Store parameters
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        
        # Initialize weights
        self._initialize_weights()
        
        # Readout weights (to be trained)
        self.W_out = None
    
    def _initialize_weights(self):
        """Initialize input and reservoir weights"""
        # Input weights
        self.W_in = np.random.uniform(-self.input_scaling, self.input_scaling, 
                                     (self.n_reservoir, self.n_inputs))
        
        # Reservoir weights (sparse random matrix)
        W = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # Apply sparsity
        mask = np.random.rand(self.n_reservoir, self.n_reservoir) < self.sparsity
        self.W = W * mask
        
        # Scale to desired spectral radius
        radius = np.max(np.abs(np.linalg.eigvals(self.W)))
        self.W *= self.spectral_radius / radius
    
    def _update(self, state, input_pattern):
        """Update reservoir state"""
        # Calculate activation
        preactivation = np.dot(self.W, state) + np.dot(self.W_in, input_pattern)
        # Apply leaky integration
        return (1 - self.leaking_rate) * state + self.leaking_rate * np.tanh(preactivation)
    
    def fit(self, inputs, outputs, washout=0, reg_param=1e-6):
        """
        Train the ESN
        
        Parameters:
        - inputs: Input features [n_samples, n_features]
        - outputs: Target outputs [n_samples, n_outputs]
        - washout: Number of initial timesteps to discard
        - reg_param: Regularization parameter for ridge regression
        """
        n_samples = inputs.shape[0]
        
        # Initialize reservoir state
        state = np.zeros((self.n_reservoir, 1))
        
        # Collection matrices
        X = np.zeros((n_samples - washout, 1 + self.n_inputs + self.n_reservoir))
        Y = np.zeros((n_samples - washout, self.n_outputs))
        
        # Run reservoir with the data and collect states
        for t in range(n_samples):
            # Update state
            current_input = inputs[t].reshape((-1, 1))
            state = self._update(state, current_input)
            
            # Add noise
            state += self.noise * (np.random.rand(self.n_reservoir, 1) - 0.5)
            
            # Store state after washout period
            if t >= washout:
                # Collect [bias; input; state]
                x = np.vstack([np.ones((1, 1)), current_input, state])
                X[t - washout] = x.flatten()
                Y[t - washout] = outputs[t]
        
        # Train output weights using ridge regression
        Xt_X = np.dot(X.T, X)
        Xt_Y = np.dot(X.T, Y)
        
        # Add regularization
        reg = reg_param * np.eye(1 + self.n_inputs + self.n_reservoir)
        self.W_out = np.dot(np.linalg.inv(Xt_X + reg), Xt_Y)
        
        # Return training error
        Y_pred = np.dot(X, self.W_out)
        mse = np.mean((Y - Y_pred) ** 2)
        return mse
    
    def predict(self, inputs, continuation=False, initial_state=None):
        """
        Predict using the trained ESN
        
        Parameters:
        - inputs: Input features [n_samples, n_features]
        - continuation: Whether to continue from last state
        - initial_state: Initial reservoir state (if not continuing)
        """
        n_samples = inputs.shape[0]
        
        # Initialize state
        if continuation and hasattr(self, 'last_state'):
            state = self.last_state
        elif initial_state is not None:
            state = initial_state
        else:
            state = np.zeros((self.n_reservoir, 1))
        
        Y_pred = np.zeros((n_samples, self.n_outputs))
        
        # Run reservoir with the data and collect predictions
        for t in range(n_samples):
            # Update state
            current_input = inputs[t].reshape((-1, 1))
            state = self._update(state, current_input)
            
            # Collect [bias; input; state]
            x = np.vstack([np.ones((1, 1)), current_input, state])
            
            # Predict output
            y_pred = np.dot(x.T, self.W_out)
            Y_pred[t] = y_pred
        
        # Store last state for possible continuation
        self.last_state = state
        
        return Y_pred

# ----- 3. Data Preparation Functions -----

def prepare_sequences(features, labels, seq_length=10):
    """
    Prepare sequences for the ESN
    
    Parameters:
    - features: Extracted features [n_samples, n_features]
    - labels: Class labels [n_samples]
    - seq_length: Length of sequences
    
    Returns:
    - X: Sequence features [n_sequences, seq_length, n_features]
    - y: Sequence labels [n_sequences]
    """
    X, y = [], []
    
    for i in range(len(features) - seq_length + 1):
        X.append(features[i:i + seq_length])
        # Use majority class in sequence
        y.append(np.bincount(labels[i:i + seq_length].astype(int)).argmax())
    
    return np.array(X), np.array(y)

def create_one_hot(labels, n_classes):
    """Convert labels to one-hot encoding"""
    one_hot = np.zeros((len(labels), n_classes))
    for i, label in enumerate(labels):
        one_hot[i, int(label)] = 1
    return one_hot

# ----- 4. Main Pipeline for Gesture Recognition -----

def main():
    """Main function for EMG-based gesture recognition"""
    # 1. Load data
    print("Loading data...")
    grasp_data = pd.read_csv('EMG_Hand/lookup.csv')
    pinch_data = pd.read_csv('EMG_Hand/lookup.csv')
    lookup_data = pd.read_csv('EMG_Hand/lookup.csv')
    
    # 2. Preprocess EMG signals
    print("Preprocessing EMG signals...")
    _, grasp_processed = preprocess_emg(grasp_data)
    _, pinch_processed = preprocess_emg(pinch_data)
    _, lookup_processed = preprocess_emg(lookup_data)
    
    # 3. Segment signals
    print("Segmenting signals...")
    grasp_segments, _ = segment_signal(grasp_data['Time[s]'].values, grasp_processed)
    pinch_segments, _ = segment_signal(pinch_data['Time[s]'].values, pinch_processed)
    lookup_segments, _ = segment_signal(lookup_data['Time[s]'].values, lookup_processed)
    
    # 4. Extract features
    print("Extracting features...")
    grasp_features = extract_features(grasp_segments)
    pinch_features = extract_features(pinch_segments)
    lookup_features = extract_features(lookup_segments)
    
    # 5. Create labels (0: grasp, 1: pinch, 2: lookup)
    grasp_labels = np.zeros(len(grasp_features))
    pinch_labels = np.ones(len(pinch_features))
    lookup_labels = np.full(len(lookup_features), 2)
    
    # 6. Combine data
    all_features = np.vstack((grasp_features, pinch_features, lookup_features))
    all_labels = np.concatenate((grasp_labels, pinch_labels, lookup_labels))
    
    # 7. Normalize features
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    # 8. Prepare sequences for ESN
    seq_length = 15  # Adjust based on gesture duration
    X, y = prepare_sequences(all_features_scaled, all_labels, seq_length)
    
    # 9. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 10. Create one-hot encoded targets
    n_gestures = 3  # grasp, pinch, lookup
    y_train_one_hot = create_one_hot(y_train, n_gestures)
    
    # 11. Flatten sequences for ESN input
    X_train_flat = X_train.reshape(-1, X_train.shape[2])
    
    # Repeat targets for each timestep in sequences
    y_train_expanded = np.repeat(y_train_one_hot, seq_length, axis=0)
    
    # 12. Create and train ESN
    print("Training ESN model...")
    esn = EchoStateNetwork(
        n_inputs=X_train.shape[2],
        n_outputs=n_gestures,
        n_reservoir=300,
        spectral_radius=0.9,
        sparsity=0.1,
        noise=0.001,
        input_scaling=1.0,
        leaking_rate=0.3
    )
    
    # Train the model
    mse = esn.fit(X_train_flat, y_train_expanded, washout=0, reg_param=1e-6)
    print(f"Training MSE: {mse:.6f}")
    
    # 13. Evaluate model
    print("Evaluating model...")
    # Flatten test sequences
    X_test_flat = X_test.reshape(-1, X_test.shape[2])
    
    # Make predictions
    y_pred_flat = esn.predict(X_test_flat)
    
    # Reshape predictions to match sequences
    y_pred_seq = y_pred_flat.reshape(X_test.shape[0], seq_length, n_gestures)
    
    # Average predictions across sequence
    y_pred_avg = np.mean(y_pred_seq, axis=1)
    y_pred_class = np.argmax(y_pred_avg, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_class)
    conf_mat = confusion_matrix(y_test, y_pred_class)
    report = classification_report(y_test, y_pred_class)
    
    # Print results
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_mat)
    print("\nClassification Report:")
    print(report)
    
    # 14. Visualize results
    plt.figure(figsize=(12, 8))
    
    # Sample of original signals
    plt.subplot(3, 2, 1)
    plt.plot(grasp_data['Time[s]'].values[:1000], grasp_processed[:1000])
    plt.title('Grasp Signal')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 2, 3)
    plt.plot(pinch_data['Time[s]'].values[:1000], pinch_processed[:1000])
    plt.title('Pinch Signal')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 2, 5)
    plt.plot(lookup_data['Time[s]'].values[:1000], lookup_processed[:1000])
    plt.title('Lookup Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Confusion matrix visualization
    plt.subplot(3, 2, 2)
    plt.imshow(conf_mat, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    gestures = ['Grasp', 'Pinch', 'Lookup']
    tick_marks = np.arange(len(gestures))
    plt.xticks(tick_marks, gestures, rotation=45)
    plt.yticks(tick_marks, gestures)
    
    # Add text annotations
    thresh = conf_mat.max() / 2
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, format(conf_mat[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_mat[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Example predictions
    plt.subplot(3, 2, 4)
    plt.bar(range(len(y_test[:20])), y_test[:20], label='True')
    plt.bar(range(len(y_pred_class[:20])), y_pred_class[:20], alpha=0.5, label='Predicted')
    plt.xticks(range(len(y_test[:20])))
    plt.xlabel('Sample')
    plt.ylabel('Class')
    plt.title('Prediction Examples')
    plt.legend()
    
    # Feature importance (using output weights)
    if hasattr(esn, 'W_out'):
        plt.subplot(3, 2, 6)
        # Get average absolute weight per feature across all classes
        feat_names = ['MAV', 'RMS', 'WL', 'ZCR', 'SSC', 'IEMG']
        feat_weights = np.mean(np.abs(esn.W_out[1:7]), axis=1)
        plt.bar(range(len(feat_weights)), feat_weights)
        plt.xticks(range(len(feat_weights)), feat_names)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
    
    plt.tight_layout()
    plt.show()

# Run the main function if executed as script
if __name__ == "__main__":
    main()