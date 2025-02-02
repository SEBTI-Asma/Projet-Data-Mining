import pandas as pd
from scipy.stats import zscore
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QMessageBox

class DataNormalizationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Data Normalization")
        self.setGeometry(200, 200, 400, 200)
        self.data = None

        # Layout for the data normalization options
        layout = QVBoxLayout()

        # Min-Max normalization button
        self.min_max_button = QPushButton("Min-Max Normalization")
        self.min_max_button.clicked.connect(self.apply_min_max_normalization)
        layout.addWidget(self.min_max_button)

        # Z-score normalization button
        self.zscore_button = QPushButton("Z-score Normalization")
        self.zscore_button.clicked.connect(self.apply_zscore_normalization)
        layout.addWidget(self.zscore_button)

        # Set dialog layout
        self.setLayout(layout)

    def set_data(self, data):
        self.data = data

    def apply_min_max_normalization(self):
        if self.parent.full_data is not None:
            # Exclude columns named 'time', 'latitude', 'longitude' for Min-Max normalization
            columns_to_normalize = self.parent.full_data.columns.difference(['time', 'latitude', 'longitude', 'geometry'])
            data_normalized = self.parent.full_data.copy()
            non_normalized_columns = []

            for column in columns_to_normalize:
                min_value = data_normalized[column].min()
                max_value = data_normalized[column].max()
                if max_value - min_value > 1e-6:  # Ensure there's a significant range for normalization
                    data_normalized[column] = (data_normalized[column] - min_value) / (max_value - min_value)
                else:
                    # Add to list of columns that were not normalized
                    non_normalized_columns.append(column)

            # Update the parent data with the normalized data
            self.parent.full_data = data_normalized
            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()

            # Show results message
            if non_normalized_columns:
                QMessageBox.warning(self, "Partial Normalization",
                                    f"Data has been normalized using Min-Max.\n"
                                    f"The following columns were not normalized due to insufficient range: {', '.join(non_normalized_columns)}")
            else:
                QMessageBox.information(self, "Normalization Complete", "Data has been normalized using Min-Max.")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def apply_zscore_normalization(self):
        if self.parent.full_data is not None:
            # Exclude columns named 'time', 'latitude', 'longitude' for Z-score normalization
            columns_to_normalize = self.parent.full_data.columns.difference(['time', 'latitude', 'longitude', 'geometry'])
            data_normalized = self.parent.full_data.copy()
            non_normalized_columns = []

            for column in columns_to_normalize:
                if data_normalized[column].std() > 1e-6:  # Check if std deviation is above threshold
                    data_normalized[column] = zscore(data_normalized[column])
                else:
                    # Add to list of columns that were not normalized
                    non_normalized_columns.append(column)

            # Update the parent data with the normalized data
            self.parent.full_data = data_normalized
            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()

            # Show results message
            if non_normalized_columns:
                QMessageBox.warning(self, "Partial Normalization",
                                    f"Data has been normalized using Z-score.\n"
                                    f"The following columns were not normalized due to low variance: {', '.join(non_normalized_columns)}")
            else:
                QMessageBox.information(self, "Normalization Complete", "Data has been normalized using Z-score.")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")
