# Write the code for the CNN-LSTM model
import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_channels, seq_len, num_features, num_classes, cnn_out_channels=32, cnn_kernel_size=3, lstm_hidden_size=64, lstm_num_layers=1, dropout=0.3):
        super(CNNLSTM, self).__init__()
        # 1D CNN expects input: (batch, channels, seq_len)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=cnn_out_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
            nn.Dropout(dropout)
        )
        # LSTM expects input: (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        # If input is (batch, seq_len, num_features), transpose to (batch, num_features, seq_len)
        x = x.transpose(1, 2)  # (batch, num_features, seq_len)
        x = self.cnn(x)        # (batch, cnn_out_channels, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_out_channels)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden_size)
        # Use the last time step's output for classification
        out = lstm_out[:, -1, :]    # (batch, lstm_hidden_size)
        out = self.fc(out)          # (batch, num_classes)
        return out

# Example usage:
# model = CNNLSTM(input_channels=num_features, seq_len=window_length, num_features=num_features, num_classes=2)
