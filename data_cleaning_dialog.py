from PyQt5.QtWidgets import QDialog,QInputDialog ,QVBoxLayout, QPushButton, QMessageBox
import pandas as pd

class DataCleaningDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Data Cleaning")
        self.setGeometry(200, 200, 400, 350)  # Adjust height to fit the new button

        # Layout for the data cleaning options
        layout = QVBoxLayout()
         # Add button to eliminate horizontal redundancy
        self.eliminate_horizontal_redundancy_button = QPushButton("Eliminate Vertical Redundancy")
        self.eliminate_horizontal_redundancy_button.clicked.connect(self.eliminate_horizontal_redundancy)
        layout.addWidget(self.eliminate_horizontal_redundancy_button)
                # Add button to eliminate vertical redundancy
        self.eliminate_vertical_redundancy_button = QPushButton("Eliminate Horizontal Redundancy")
        self.eliminate_vertical_redundancy_button.clicked.connect(self.eliminate_vertical_redundancy)
        layout.addWidget(self.eliminate_vertical_redundancy_button)

        # Add buttons for different outlier handling methods
        self.remove_outliers_button = QPushButton("Remove Outliers")
        self.remove_outliers_button.clicked.connect(self.remove_outliers)
        layout.addWidget(self.remove_outliers_button)

        self.replace_outliers_mean_button = QPushButton("Replace Outliers with Mean")
        self.replace_outliers_mean_button.clicked.connect(self.replace_outliers_with_mean)
        layout.addWidget(self.replace_outliers_mean_button)
    # Add button for replacing outliers with contextual mean
        self.replace_outliers_contextual_mean_button = QPushButton("Replace Outliers with Contextual Mean")
        self.replace_outliers_contextual_mean_button.clicked.connect(self.replace_outliers_with_contextual_mean)
        layout.addWidget(self.replace_outliers_contextual_mean_button)
    #Add button for replacing outliers with contextual mean
        self.replace_outliers_with_IQR_button = QPushButton("Cap Outliers with IQR Limit")
        self.replace_outliers_with_IQR_button.clicked.connect(self.replace_outliers_with_iqr_limits)
        layout.addWidget(self.replace_outliers_with_IQR_button)
          # Add button to delete a column
        self.delete_column_button = QPushButton("Delete a Column")
        self.delete_column_button.clicked.connect(self.delete_column)
        layout.addWidget(self.delete_column_button)

        # Add button to delete a row
        self.delete_row_button = QPushButton("Delete a Row")
        self.delete_row_button.clicked.connect(self.delete_row)
        layout.addWidget(self.delete_row_button)


        self.replace_outliers_median_button = QPushButton("Replace Outliers with Median")
        self.replace_outliers_median_button.clicked.connect(self.replace_outliers_with_median)
        layout.addWidget(self.replace_outliers_median_button)

        # Add button to remove rows with only zeroes
        self.remove_zero_rows_button = QPushButton("Remove Rows with Only Zeroes")
        self.remove_zero_rows_button.clicked.connect(self.remove_zero_rows)
        layout.addWidget(self.remove_zero_rows_button)

        self.cap_outliers_button = QPushButton("Cap Outliers")
        self.cap_outliers_button.clicked.connect(self.cap_outliers)
        layout.addWidget(self.cap_outliers_button)
        self.replace_zero_rows_with_mean_button = QPushButton("Replace Zero Rows with Mean")
        self.replace_zero_rows_with_mean_button.clicked.connect(self.replace_zero_rows_with_mean)
        layout.addWidget(self.replace_zero_rows_with_mean_button)
        self.replace_zero_rows_with_median_button = QPushButton("Replace Zero Rows with Median")
        self.replace_zero_rows_with_median_button.clicked.connect(self.replace_zero_rows_with_median)
        layout.addWidget(self.replace_zero_rows_with_median_button)
        # Add button to remove rows with NaN values
        self.remove_nan_button = QPushButton("Remove Rows with NaN")
        self.remove_nan_button.clicked.connect(self.remove_nan_rows)
        layout.addWidget(self.remove_nan_button)
        self.remove_low_variance_button = QPushButton("Remove Rows low variance features")
        self.remove_low_variance_button.clicked.connect(self.remove_low_variance_features)
        layout.addWidget(self.remove_low_variance_button)
        self.setLayout(layout)

    def remove_low_variance_features(self, threshold=0.01):
        """
        Removes columns that have 'low variance' based on a ratio of:
            std / (max - min) < threshold
        
        Parameters
        ----------
        threshold : float, optional
            The maximum ratio of std/(max-min) considered "low variance".
            Default is 0.01 (1%).
        """
        if self.parent.full_data is not None:
            # Save current state before modifying
            self.parent.save_current_state()

            # Identify numeric columns
            numeric_columns = self.parent.full_data.select_dtypes(include=['float64', 'int64']).columns
            
            low_variance_cols = []
            
            for col in numeric_columns:
                col_data = self.parent.full_data[col]
                col_min, col_max = col_data.min(), col_data.max()
                col_range = col_max - col_min
                
                # Avoid division by zero if all values are the same (range=0)
                if col_range == 0:
                    # If range=0, the column is constant (std=0). Usually worth dropping as "low variance."
                    low_variance_cols.append(col)
                else:
                    col_std = col_data.std()
                    ratio = col_std / col_range
                    # Check if ratio is below threshold
                    if ratio < threshold:
                        low_variance_cols.append(col)
            
            # Drop the identified low variance columns
            if low_variance_cols:
                self.parent.full_data.drop(columns=low_variance_cols, inplace=True)
            
            # Refresh the data display
            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()

            if low_variance_cols:
                QMessageBox.information(
                    self,
                    "Low Variance Columns Removed",
                    f"The following columns were removed due to low variance:\n{', '.join(low_variance_cols)}"
                )
            else:
                QMessageBox.information(
                    self,
                    "No Columns Removed",
                    "No columns were found with variance below the chosen threshold."
                )
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")
    
    def calculate_outliers(self):
        """Calculate outliers based on IQR for each numeric column."""
        columns_to_check = self.parent.full_data.select_dtypes(include=['float64', 'int64']).columns
        Q1 = self.parent.full_data[columns_to_check].quantile(0.25)
        Q3 = self.parent.full_data[columns_to_check].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (self.parent.full_data[columns_to_check] < (Q1 - 1.5 * IQR)) | (self.parent.full_data[columns_to_check] > (Q3 + 1.5 * IQR))
        return outliers

    def remove_outliers(self):
        """Removes rows with outliers based on IQR."""
        if self.parent.full_data is not None:
            outliers = self.calculate_outliers()
            self.parent.full_data = self.parent.full_data[~outliers.any(axis=1)]
            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()
            QMessageBox.information(self, "Outliers Removed", f"Outliers removed. Remaining rows: {len(self.parent.full_data)}.")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def replace_outliers_with_contextual_mean(self):
        """Replace outlier values with the mean of the same season and nearby coordinates, excluding outliers."""
        if self.parent.full_data is not None:
            # Ensure the dataset contains required columns
            if 'time' not in self.parent.full_data.columns or 'latitude' not in self.parent.full_data.columns or 'longitude' not in self.parent.full_data.columns:
                QMessageBox.warning(self, "Missing Columns", "The dataset must include 'date', 'latitude', and 'longitude' columns.")
                return

            # Add a 'season' column to the dataset
            self.parent.full_data['time'] = pd.to_datetime(self.parent.full_data['time'])
            self.parent.full_data['season'] = self.parent.full_data['time'].dt.month % 12 // 3 + 1  # Map months to seasons (1=Winter, etc.)

            # Identify outliers for each numeric column
            outliers = self.calculate_outliers()

            for column in outliers.columns:
                for index, is_outlier in outliers[column].items():
                    if is_outlier:
                        # Get the current row's season, latitude, and longitude
                        row = self.parent.full_data.loc[index]
                        season = row['season']
                        latitude = row['latitude']
                        longitude = row['longitude']

                        # Define thresholds for "nearby" coordinates
                        latitude_threshold = 0.5
                        longitude_threshold = 0.5

                        # Filter rows for the same season and nearby coordinates
                        nearby_data = self.parent.full_data[
                            (self.parent.full_data['season'] == season) &
                            (abs(self.parent.full_data['latitude'] - latitude) <= latitude_threshold) &
                            (abs(self.parent.full_data['longitude'] - longitude) <= longitude_threshold)
                        ]

                        # Exclude outliers from the nearby data
                        valid_data = nearby_data[~outliers.loc[nearby_data.index, column]]

                        # Calculate the mean from the valid data
                        contextual_mean = valid_data[column].mean()

                        if not pd.isna(contextual_mean):  # Only replace if a valid mean is computed
                            self.parent.full_data.at[index, column] = contextual_mean

            # Refresh the data display
            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()
            QMessageBox.information(self, "Outliers Replaced", "Outliers have been replaced with the contextual mean.")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def eliminate_horizontal_redundancy(self):
        """Eliminates columns with a correlation greater than or equal to 0.95."""
        if self.parent.full_data is not None:
            numeric_columns = self.parent.full_data.select_dtypes(include=['float64', 'int64']).columns

            # Compute the correlation matrix
            correlation_matrix = self.parent.full_data[numeric_columns].corr()

            # Find pairs of columns with correlation >= 0.95
            columns_to_drop = set()
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] >= 0.90:
                        column_to_drop = correlation_matrix.columns[j]
                        columns_to_drop.add(column_to_drop)

            # Drop redundant columns
            self.parent.full_data.drop(columns=list(columns_to_drop), inplace=True)

            # Refresh the data display
            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()

            # Create a message indicating which columns were removed
            if columns_to_drop:
                removed_columns = ", ".join(columns_to_drop)
                QMessageBox.information(
                    self, "Horizontal Redundancy Eliminated",
                    f"The following columns with high correlation (≥ 0.95) have been removed: {removed_columns}"
                )
            else:
                QMessageBox.information(
                    self, "No Redundancy Found",
                    "No columns with high correlation (≥ 0.95) were found."
                )
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def remove_zero_rows(self):
        """
        Removes rows where all numeric values are zero, 
        except in the columns 'latitude', 'longitude', 'geometry', and 'time'.
        """
        if self.parent.full_data is not None:
            self.parent.save_current_state()
            # Define the columns to exclude from the zero check
            excluded_columns = ['latitude', 'longitude', 'geometry', 'time']
            excluded_columns = [col for col in excluded_columns if col in self.parent.full_data.columns]

            # Identify numeric columns to check
            numeric_columns = self.parent.full_data.select_dtypes(include=['float64', 'int64']).columns
            columns_to_check = [col for col in numeric_columns if col not in excluded_columns]

            # Create a boolean mask for rows with all zeroes in the specified columns
            mask = (self.parent.full_data[columns_to_check] == 0).all(axis=1)

            # Count rows to remove
            rows_to_remove = mask.sum()

            # Remove rows with all zeroes
            self.parent.full_data = self.parent.full_data[~mask]

            # Reset the index of the DataFrame
            self.parent.full_data.reset_index(drop=True, inplace=True)

            # Refresh the data display
            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()

            # Notify the user
            QMessageBox.information(
                self, "Zero Rows Removed",
                f"{rows_to_remove} rows with only zeroes (excluding latitude, longitude, geometry, and time) have been removed."
            )
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")




    def delete_column(self):
        """
        Deletes a column from the dataset.
        """
        if self.parent.full_data is not None:
            column, ok = QInputDialog.getItem(
                self, "Delete Column", "Select a column to delete:", self.parent.full_data.columns.tolist(), 0, False
            )
            if ok and column:
                self.parent.save_current_state()
                self.parent.full_data.drop(columns=[column], inplace=True)
                self.parent.current_page = 0
                self.parent.update_total_pages()
                self.parent.display_data()
                QMessageBox.information(self, "Column Deleted", f"The column '{column}' has been deleted.")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def delete_row(self):
        """
        Deletes a specific row from the dataset.
        """
        if self.parent.full_data is not None:
            row_index, ok = QInputDialog.getInt(
                self, "Delete Row", "Enter the row index to delete:", 0, 0, len(self.parent.full_data) - 1
            )
            if ok:
                self.parent.full_data.drop(index=row_index, inplace=True)
                self.parent.full_data.reset_index(drop=True, inplace=True)
                self.parent.current_page = 0
                self.parent.update_total_pages()
                self.parent.display_data()
                QMessageBox.information(self, "Row Deleted", f"The row at index {row_index} has been deleted.")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def replace_outliers_with_mean(self):
        """Replaces outlier values with the column mean."""
        if self.parent.full_data is not None:
            outliers = self.calculate_outliers()
            for column in outliers.columns:
                mean_value = self.parent.full_data[column].mean()
                self.parent.full_data.loc[outliers[column], column] = mean_value

            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()
            QMessageBox.information(self, "Outliers Replaced", "Outliers have been replaced with the column mean.")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def replace_outliers_with_median(self):
        """Replaces outlier values with the column median."""
        if self.parent.full_data is not None:
            outliers = self.calculate_outliers()
            for column in outliers.columns:
                median_value = self.parent.full_data[column].median()
                self.parent.full_data.loc[outliers[column], column] = median_value

            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()
            QMessageBox.information(self, "Outliers Replaced", "Outliers have been replaced with the column median.")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def cap_outliers(self):
        """Caps outliers to the 5th and 95th percentiles."""
        if self.parent.full_data is not None:
            columns_to_check = self.parent.full_data.select_dtypes(include=['float64', 'int64']).columns
            lower_bound = self.parent.full_data[columns_to_check].quantile(0.05)
            upper_bound = self.parent.full_data[columns_to_check].quantile(0.95)

            for column in columns_to_check:
                self.parent.full_data[column] = self.parent.full_data[column].clip(lower=lower_bound[column], upper=upper_bound[column])

            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()
            QMessageBox.information(self, "Outliers Capped", "Outliers have been capped to the 5th and 95th percentiles.")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def remove_nan_rows(self):
        """Removes rows containing NaN values."""
        if self.parent.full_data is not None:
            initial_rows = len(self.parent.full_data)
            self.parent.full_data = self.parent.full_data.dropna()
            final_rows = len(self.parent.full_data)
            removed_rows = initial_rows - final_rows

            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()
            QMessageBox.information(self, "NaN Rows Removed", f"{removed_rows} rows with NaN values have been removed.")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def replace_zero_rows_with_mean(self):
        """
        Replaces rows with only zero values with the column mean,
        excluding certain columns ('latitude', 'longitude', 'geometry', 'time') 
        and excluding zero rows from mean calculations.
        """
        if self.parent.full_data is not None:
            # Save the current state of the dataset
            self.parent.save_current_state()

            # Define excluded columns
            excluded_columns = ['latitude', 'longitude', 'geometry', 'time']
            excluded_columns = [col for col in excluded_columns if col in self.parent.full_data.columns]

            # Identify numeric columns to check
            numeric_columns = self.parent.full_data.select_dtypes(include=['float64', 'int64']).columns
            columns_to_process = [col for col in numeric_columns if col not in excluded_columns]

            # Create a mask for rows with all zeroes in the specified columns
            zero_row_mask = (self.parent.full_data[columns_to_process] == 0).all(axis=1)

            # Compute column means excluding zero rows
            valid_data = self.parent.full_data[~zero_row_mask]
            column_means = valid_data[columns_to_process].mean()

            # Replace zero rows with column means
            for index in self.parent.full_data[zero_row_mask].index:
                self.parent.full_data.loc[index, columns_to_process] = column_means

            # Refresh the data display
            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()

            # Notify the user
            QMessageBox.information(
                self, "Zero Rows Replaced",
                f"{zero_row_mask.sum()} rows with only zero values have been replaced with the column mean."
            )
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")
    def replace_zero_rows_with_median(self):
        """
        Replaces rows with only zero values with the column mean,
        excluding certain columns ('latitude', 'longitude', 'geometry', 'time') 
        and excluding zero rows from mean calculations.
        """
        if self.parent.full_data is not None:
            # Save the current state of the dataset
            self.parent.save_current_state()

            # Define excluded columns
            excluded_columns = ['latitude', 'longitude', 'geometry', 'time']
            excluded_columns = [col for col in excluded_columns if col in self.parent.full_data.columns]

            # Identify numeric columns to check
            numeric_columns = self.parent.full_data.select_dtypes(include=['float64', 'int64']).columns
            columns_to_process = [col for col in numeric_columns if col not in excluded_columns]

            # Create a mask for rows with all zeroes in the specified columns
            zero_row_mask = (self.parent.full_data[columns_to_process] == 0).all(axis=1)

            # Compute column means excluding zero rows
            valid_data = self.parent.full_data[~zero_row_mask]
            column_means = valid_data[columns_to_process].median()

            # Replace zero rows with column means
            for index in self.parent.full_data[zero_row_mask].index:
                self.parent.full_data.loc[index, columns_to_process] = column_means

            # Refresh the data display
            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()

            # Notify the user
            QMessageBox.information(
                self, "Zero Rows Replaced",
                f"{zero_row_mask.sum()} rows with only zero values have been replaced with the column mean."
            )
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def replace_outliers_with_iqr_limits(self):
        """
        Caps outlier values based on IQR rule for numeric columns,
        excluding columns like 'time', 'longitude', 'latitude', etc.
        """
        if self.parent.full_data is not None:
            # Define which columns we want to exclude from outlier processing
            excluded_columns = ['time', 'longitude', 'latitude']
            # You can add any other columns you don't want to process here.

            # Identify numeric columns to process, excluding the above
            columns_to_check = self.parent.full_data.select_dtypes(include=['float64', 'int64']).columns
            columns_to_check = [col for col in columns_to_check if col not in excluded_columns]
            
            # Calculate IQR bounds and cap each column
            for column in columns_to_check:
                Q1 = self.parent.full_data[column].quantile(0.25)
                Q3 = self.parent.full_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Clip values outside [lower_bound, upper_bound]
                self.parent.full_data[column] = self.parent.full_data[column].clip(
                    lower=lower_bound, 
                    upper=upper_bound
                )

            # Refresh display in your PyQt interface
            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()

            # Notify user
            QMessageBox.information(
                self, 
                "Outliers Replaced (IQR)", 
                "Outliers have been capped to the IQR-based lower and upper limits."
            )
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")
    
        
    def eliminate_vertical_redundancy(self):
            """
            Eliminates duplicate rows in the dataset while excluding certain columns (e.g., 'time', 'latitude', 'longitude', 'geometry') 
            from the redundancy check.
            """
            if self.parent.full_data is not None:
                excluded_columns = ['time', 'latitude', 'longitude', 'geometry']

                # Ensure the excluded columns exist in the dataset
                excluded_columns = [col for col in excluded_columns if col in self.parent.full_data.columns]

                # Identify the columns to include in redundancy checks
                columns_to_check = [col for col in self.parent.full_data.columns if col not in excluded_columns]

                # Drop duplicate rows based on the selected columns
                initial_row_count = len(self.parent.full_data)
                self.parent.full_data = self.parent.full_data.drop_duplicates(subset=columns_to_check)
                final_row_count = len(self.parent.full_data)

                # Refresh the data display
                self.parent.current_page = 0
                self.parent.update_total_pages()
                self.parent.display_data()

                # Notify the user
                removed_rows = initial_row_count - final_row_count
                QMessageBox.information(
                    self, "Vertical Redundancy Eliminated",
                    f"{removed_rows} duplicate rows have been removed."
                )
            else:
                QMessageBox.warning(self, "No Data", "Please import a dataset first.")
