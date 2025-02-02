from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QMessageBox, QInputDialog
import pandas as pd

class DataDiscretizationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Data Discretization")
        self.setGeometry(200, 200, 400, 200)

        # Layout for the data discretization options
        layout = QVBoxLayout()

        # Buttons for different discretization methods
        self.equal_width_button = QPushButton("Equal Width Discretization (Very Low - Very High)")
        self.equal_width_button.clicked.connect(self.equal_width_discretization)
        layout.addWidget(self.equal_width_button)

        self.equal_frequency_button = QPushButton("Equal Frequency Discretization (Very Low - Very High)")
        self.equal_frequency_button.clicked.connect(self.equal_frequency_discretization)
        layout.addWidget(self.equal_frequency_button)


        # Button to apply the transformation
        self.transform_button = QPushButton("Transform Intervals to Categories")
        self.transform_button.clicked.connect(self.transform_intervals_to_categories)
        layout.addWidget(self.transform_button)

        # Set dialog layout
        self.setLayout(layout)

    def get_column_options(self):
        """
        Generate a list of columns for discretization.
        Excludes 'latitude', 'longitude', and 'geometry' columns, but includes an 'All Numerical Columns' option.
        """
        if self.parent.full_data is not None:
            numerical_columns = self.parent.full_data.select_dtypes(include=['number']).columns.tolist()
            # Exclude specific columns
            excluded_columns = {'latitude', 'longitude', 'geometry'}
            valid_columns = [col for col in numerical_columns if col not in excluded_columns]
            return ["All Numerical Columns"] + valid_columns
        return []

    def apply_discretization(self, column, method, bins, labels, display_intervals=False):
        """
        Apply the specified discretization method to the given column(s).
        """
        updated_data = self.parent.full_data.copy()  # Copy the existing DataFrame for modifications

        if column == "All Numerical Columns":
            # Apply discretization to all numerical columns (excluding latitude, longitude, geometry)
            for col in self.get_column_options()[1:]:
                try:
                    # Check if the column contains only zeros
                    if updated_data[col].nunique() == 1:
                        updated_data[col] = "No Data"  # Assign default category
                        continue

                    if method == "equal_width":
                        discretized = pd.cut(updated_data[col], bins=bins, duplicates="drop")
                    elif method == "equal_frequency":
                        discretized = pd.qcut(updated_data[col], q=bins, duplicates="drop")

                    # Map to intervals or categorical labels
                    if display_intervals:
                        updated_data[col] = discretized
                    else:
                        updated_data[col] = discretized.apply(lambda x: labels[discretized.cat.categories.tolist().index(x)])
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"An error occurred while discretizing '{col}': {e}")
        else:
            # Apply discretization to a single column
            try:
                # Check if the column contains only zeros
                if updated_data[column].nunique() == 1:
                    updated_data[column] = "No Data"  # Assign default category
                    QMessageBox.information(
                        self, "Discretization Complete",
                        f"The column '{column}' contains only zeros and has been categorized as 'No Data'."
                    )
                    self.parent.full_data = updated_data  # Update the dataset
                    self.parent.display_data()  # Refresh the display
                    return

                if method == "equal_width":
                    discretized = pd.cut(updated_data[column], bins=bins, duplicates="drop")
                elif method == "equal_frequency":
                    discretized = pd.qcut(updated_data[column], q=bins, duplicates="drop")

                # Map to intervals or categorical labels
                if display_intervals:
                    updated_data[column] = discretized
                else:
                    updated_data[column] = discretized.apply(lambda x: labels[discretized.cat.categories.tolist().index(x)])
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while discretizing '{column}': {e}")

        # Update the parent table display with the modified data
        self.parent.full_data = updated_data  # Replace the full_data with the updated one
        self.parent.display_data()  # Refresh the displayed data to show the new table



    def equal_width_discretization(self):
        if self.parent.full_data is not None:
            column, ok = QInputDialog.getItem(
                self,
                "Select Column",
                "Choose column for equal width discretization:",
                self.get_column_options(),
                0,
                False
            )
            if ok:
                # Ask the user if they want intervals or categorical labels
                display_intervals, ok = QInputDialog.getItem(
                    self,
                    "Display Format",
                    "Choose display format:",
                    ["Intervals", "Categorical Labels (Very Low to Very High)"],
                    0,
                    False
                )
                if ok:
                    try:
                        bins = 5  # Fixed number of bins
                        labels = ["Very Low", "Low", "Medium", "High", "Very High"]
                        self.apply_discretization(
                            column,
                            method="equal_width",
                            bins=bins,
                            labels=labels,
                            display_intervals=(display_intervals == "Intervals")
                        )
                        QMessageBox.information(self, "Discretization Complete", "Equal width discretization applied.")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"An error occurred during discretization: {e}")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")
    def transform_intervals_to_categories(self):
        """
        Transforms interval data in a dataset into categorical data.
        """
        if self.parent.full_data is not None:
            # Copy the existing dataset to avoid modifying the original directly
            data = self.parent.full_data.copy()

            # Define columns to exclude from the transformation
            excluded_columns = {'latitude', 'longitude', 'time', 'geometry'}
            transformable_columns = [col for col in data.columns if col not in excluded_columns]

            for column in transformable_columns:
                # Check if the column contains intervals (e.g., "(0.0, 1.0]")
                if pd.api.types.is_categorical_dtype(data[column]):
                    try:
                        # Map intervals to categories
                        new_categories = [
                            f"very_low_{column}" if i == 0 else
                            f"low_{column}" if i == 1 else
                            f"medium_{column}" if i == 2 else
                            f"high_{column}" if i == 3 else
                            f"very_high_{column}"
                            for i in range(len(data[column].cat.categories))
                        ]
                        data[column] = data[column].cat.codes.map(lambda x: new_categories[x])

                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"An error occurred while transforming '{column}': {e}")

            # Update the parent dataset and refresh the display
            self.parent.full_data = data
            self.parent.display_data()
            QMessageBox.information(self, "Transformation Complete", "Intervals have been transformed to categories.")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def equal_frequency_discretization(self):
        if self.parent.full_data is not None:
            column, ok = QInputDialog.getItem(
                self,
                "Select Column",
                "Choose column for equal frequency discretization:",
                self.get_column_options(),
                0,
                False
            )
            if ok:
                # Ask the user if they want intervals or categorical labels
                display_intervals, ok = QInputDialog.getItem(
                    self,
                    "Display Format",
                    "Choose display format:",
                    ["Intervals", "Categorical Labels (Very Low to Very High)"],
                    0,
                    False
                )
                if ok:
                    try:
                        bins = 5  # Fixed number of bins
                        labels = ["Very Low", "Low", "Medium", "High", "Very High"]
                        self.apply_discretization(
                            column,
                            method="equal_frequency",
                            bins=bins,
                            labels=labels,
                            display_intervals=(display_intervals == "Intervals")
                        )
                        QMessageBox.information(self, "Discretization Complete", "Equal frequency discretization applied.")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"An error occurred during discretization: {e}")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")
