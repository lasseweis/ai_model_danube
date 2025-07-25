# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_model(input_shape, lstm_units, dropout_rate, learning_rate):
    """
    Builds and compiles the LSTM model.
    
    Args:
        input_shape (tuple): The shape of the input data (look_back_window, num_features).
        lstm_units (int): Number of neurons in the LSTM layer.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        keras.Model: The compiled LSTM model.
    """
    model = Sequential()
    
    # First LSTM layer with dropout. return_sequences=True is needed if another LSTM layer follows.
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_rate))
    
    # Output layer: A Dense layer with one neuron to predict the discharge value.
    model.add(Dense(units=1))
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model