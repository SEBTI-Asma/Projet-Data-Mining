from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QScrollArea, QWidget, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, QInputDialog, QApplication
from PyQt5.QtGui import QClipboard
import pandas as pd

class DataManipulationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Data Manipulation")
        self.setGeometry(150, 150, 600, 400)

        layout = QVBoxLayout()
        self.import_button = QPushButton("Import Dataset")
        self.import_button.clicked.connect(self.import_data)
        layout.addWidget(self.import_button)

        self.description_button = QPushButton("Dataset Description")
        self.description_button.clicked.connect(self.show_description)
        layout.addWidget(self.description_button)

        self.update_button = QPushButton("Update an Instance")
        self.update_button.clicked.connect(self.update_instance)
        layout.addWidget(self.update_button)

        self.delete_button = QPushButton("Delete an Instance")
        self.delete_button.clicked.connect(self.delete_instance)
        layout.addWidget(self.delete_button)

        self.save_button = QPushButton("Save Dataset")
        self.save_button.clicked.connect(self.save_data)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def import_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.parent.file_path = file_path
            self.parent.full_data = pd.read_csv(file_path)
            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()
            QMessageBox.information(self, "Import Successful", "Dataset imported successfully.")
    
    def update_instance(self):
        
            if self.parent.full_data is not None:
                # Get the row index to update
                row, ok = QInputDialog.getInt(self, "Update Row", "Enter row number to update:", 0, 0, len(self.parent.full_data) - 1)
                if ok:
                    # Get the column name to update
                    column, ok = QInputDialog.getItem(self, "Update Column", "Choose column:", self.parent.full_data.columns.tolist(), 0, False)
                    if ok:
                        # Get the new value for the cell
                        new_value, ok = QInputDialog.getText(self, "New Value", f"Enter new value for {column} at row {row}:")
                        if ok:
                            # Update the dataset with the new value
                            self.parent.full_data.at[row, column] = new_value
                            self.parent.display_data()
                            QMessageBox.information(self, "Update Successful", f"Row {row} updated successfully.")
            else:
                QMessageBox.warning(self, "No Data", "Please import a dataset first.")    

    def delete_instance(self):
        """Deletes a specific row from the dataset."""
        if self.parent.full_data is not None:
            # Prompt the user to select a row to delete
            row, ok = QInputDialog.getInt(self, "Delete Row", "Enter row number to delete:", 0, 0, len(self.parent.full_data) - 1)
            if ok:
                # Drop the selected row and reset the index
                self.parent.full_data.drop(index=row, inplace=True)
                self.parent.full_data.reset_index(drop=True, inplace=True)
                self.parent.display_data()
                QMessageBox.information(self, "Delete Successful", f"Row {row} deleted successfully.")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")
    def save_data(self):
        """Saves the current dataset to a CSV file."""
        if self.parent.full_data is not None:
            # Prompt the user to select a location and name for the CSV file
            save_path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)")
            if save_path:
                # Ensure the file has a .csv extension
                if not save_path.lower().endswith('.csv'):
                    save_path += '.csv'
                try:
                    # Save the DataFrame to the specified path
                    self.parent.full_data.to_csv(save_path, index=False)
                    QMessageBox.information(self, "Save Successful", f"Dataset saved successfully to {save_path}.")
                except Exception as e:
                    QMessageBox.warning(self, "Save Error", f"An error occurred while saving the file: {e}")
            else:
                QMessageBox.warning(self, "Save Cancelled", "No file was selected for saving.")
        else:
            QMessageBox.warning(self, "No Data", "Please import or load a dataset before saving.")


    def show_description(self):
        if self.parent.full_data is not None:
            data = self.parent.full_data
            
            # Overview Table
            overview = pd.DataFrame({
                "Column Name": data.columns,
                "Data Type": data.dtypes,
                "Missing Values": data.isnull().sum(),
                "Unique Values": data.nunique()
            })
            overview['Column Name'] = overview['Column Name'].astype(str)
            overview['Data Type'] = overview['Data Type'].astype(str)
            overview_summary = f"Dataset contains {data.shape[0]} rows and {data.shape[1]} columns."

            # Statistical Summary
            numeric_data = data.select_dtypes(include=['number'])
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1

            stats = pd.DataFrame({
                "Mean": numeric_data.mean(),
                "Median": numeric_data.median(),
                "Mode": numeric_data.mode().iloc[0],
                "Std Dev": numeric_data.std(),
                "Min": numeric_data.min(),
                "Q1 (25th Percentile)": Q1,
                "Q3 (75th Percentile)": Q3,
                "Max": numeric_data.max(),
                "IQR": IQR,
                "Lower Bound (Q1 - 1.5 * IQR)": Q1 - 1.5 * IQR,
                "Upper Bound (Q3 + 1.5 * IQR)": Q3 + 1.5 * IQR
            })

            # Include the column names as part of the stats
            stats.index.name = "Column Name"
            stats.reset_index(inplace=True)

            # Creating the Scrollable Window with Tables
            self.description_window = QDialog(self)
            self.description_window.setWindowTitle("Dataset Description")
            self.description_window.setGeometry(100, 100, 900, 600)
            desc_layout = QVBoxLayout()

            # Overview Label
            overview_label = QLabel(overview_summary)
            desc_layout.addWidget(overview_label)

            # Display Overview Table
            self.overview_table = self.create_table_from_df(overview, "Overview")
            desc_layout.addWidget(self.overview_table)

            # Display Statistics Table
            self.stats_table = self.create_table_from_df(stats, "Statistics")
            desc_layout.addWidget(self.stats_table)

            # Copy button to copy table data
            copy_button = QPushButton("Copy Selected Table")
            copy_button.clicked.connect(self.copy_table)
            desc_layout.addWidget(copy_button)

            # Adding layout to scrollable area
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll_content = QWidget()
            scroll_content.setLayout(desc_layout)
            scroll.setWidget(scroll_content)

            # Setting main layout
            main_layout = QVBoxLayout()
            main_layout.addWidget(scroll)
            self.description_window.setLayout(main_layout)
            self.description_window.exec_()
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")



    def create_table_from_df(self, df, title):
        """Creates a QTableWidget from a DataFrame."""
        table = QTableWidget()
        table.setRowCount(df.shape[0])
        table.setColumnCount(df.shape[1])
        table.setHorizontalHeaderLabels(df.columns)

        # Populate the table with data
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                table.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))

        table.setWindowTitle(title)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        return table

    def copy_table(self):
        """Copies selected cells from the currently selected table to the clipboard."""
        # Determine which table has a selected range
        table = None
        if self.overview_table.selectedRanges():
            table = self.overview_table
        elif self.stats_table.selectedRanges():
            table = self.stats_table
        
        if table is None:
            QMessageBox.warning(self, "No Table Selected", "Please select a table and a range of cells to copy.")
            return

        # Collect data from the selected cells
        selected_data = ""
        for selected_range in table.selectedRanges():
            for row in range(selected_range.topRow(), selected_range.bottomRow() + 1):
                row_data = []
                for col in range(selected_range.leftColumn(), selected_range.rightColumn() + 1):
                    item = table.item(row, col)
                    row_data.append(item.text() if item else "")
                selected_data += "\t".join(row_data) + "\n"

        # Copy selected data to clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText(selected_data)
        QMessageBox.information(self, "Copy Complete", "Selected table data has been copied to the clipboard.")
