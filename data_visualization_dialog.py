import pandas as pd
import matplotlib
from seaborn import heatmap
matplotlib.use('QtAgg')

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QInputDialog, QMessageBox

def convert_intervals_to_midpoints(series):
        """
        Converts interval strings into numeric midpoints.
        Example: "(-2.0, -1.0]" → -1.5
        """
        import numpy as np
        return series.str.extract(r"\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)") \
                    .astype(float) \
                    .mean(axis=1).replace(np.nan, 0)

class DataVisualizationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Data Visualization")
        self.setGeometry(200, 200, 400, 200)

        # Layout for the data visualization options
        layout = QVBoxLayout()

        # Buttons for different types of plots
        self.boxplot_button = QPushButton("Boxplot")
        self.boxplot_button.clicked.connect(self.plot_boxplot)
        layout.addWidget(self.boxplot_button)

        self.scatter_button = QPushButton("Scatter Plot")
        self.scatter_button.clicked.connect(self.plot_scatter)
        layout.addWidget(self.scatter_button)

        self.histogram_button = QPushButton("Histogram")
        self.histogram_button.clicked.connect(self.plot_histogram)
        layout.addWidget(self.histogram_button)

        # Add a new button for the heatmap
        self.heatmap_button = QPushButton("Correlation Heatmap")
        self.heatmap_button.clicked.connect(self.plot_heatmap)
        layout.addWidget(self.heatmap_button)

        # Add button for visualizing intervals or categories
        self.barplot_button = QPushButton("Bar Plot for Categorical or Interval Data")
        self.barplot_button.clicked.connect(self.plot_barplot)
        layout.addWidget(self.barplot_button)

        self.discretized_histogram_button = QPushButton("Discretized Histogram")
        self.discretized_histogram_button.clicked.connect(self.plot_discretized_histogram)
        layout.addWidget(self.discretized_histogram_button)


        # Add buttons for advanced visualizations
        self.parallel_coordinates_button = QPushButton("Parallel Coordinates")
        self.parallel_coordinates_button.clicked.connect(self.plot_parallel_coordinates)
        layout.addWidget(self.parallel_coordinates_button)

        self.pairwise_scatter_button = QPushButton("Pairwise Scatterplot Matrix")
        self.pairwise_scatter_button.clicked.connect(self.plot_pairwise_scatter)
        layout.addWidget(self.pairwise_scatter_button)

        self.bubble_chart_button = QPushButton("Bubble Chart")
        self.bubble_chart_button.clicked.connect(self.plot_bubble_chart)
        layout.addWidget(self.bubble_chart_button)

        # Set dialog layout
        self.setLayout(layout)
    
    
    def plot_boxplot(self):
        if self.parent.full_data is not None:
            columns = self.parent.full_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if columns:
                column, ok = QInputDialog.getItem(self, "Select Column", "Choose column for boxplot:", columns, 0, False)
                if ok:
                    plt.figure(figsize=(8, 6))
                    self.parent.full_data.boxplot(column=column)
                    plt.title(f'Boxplot of {column}')
                    plt.ylabel(column)
                    plt.show()
            else:
                QMessageBox.warning(self, "No Numeric Columns", "No numeric columns available for boxplot.")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def plot_scatter(self):
        if self.parent.full_data is not None:
            columns = self.parent.full_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if len(columns) >= 2:
                col_x, ok_x = QInputDialog.getItem(self, "Select X Column", "Choose X column for scatter plot:", columns, 0, False)
                if ok_x:
                    col_y, ok_y = QInputDialog.getItem(self, "Select Y Column", "Choose Y column for scatter plot:", columns, 1, False)
                    if ok_y:
                        # Calculate Pearson correlation coefficient
                        correlation = self.parent.full_data[[col_x, col_y]].corr().iloc[0, 1]

                        # Create scatter plot
                        plt.figure(figsize=(8, 6))
                        plt.scatter(self.parent.full_data[col_x], self.parent.full_data[col_y], alpha=0.5)
                        plt.title(f'Scatter Plot of {col_x} vs {col_y}')
                        plt.xlabel(col_x)
                        plt.ylabel(col_y)

                        # Display Pearson correlation coefficient on the plot
                        plt.annotate(f'Pearson Correlation: {correlation:.2f}', 
                                     xy=(0.05, 0.95), xycoords='axes fraction', 
                                     fontsize=12, color='red', 
                                     bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white'))

                        plt.show()
            else:
                QMessageBox.warning(self, "Insufficient Columns", "Need at least two numeric columns for scatter plot.")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def plot_histogram(self):
        if self.parent.full_data is not None:
            # Check for numeric columns
            columns = self.parent.full_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if not columns:
                QMessageBox.warning(self, "No Numeric Columns", "No numeric columns available for histogram.")
                return
            
            # Check for discretized columns
            discretized_columns = self.parent.full_data.select_dtypes(include=['category', 'object']).columns.tolist()
            column, ok_col = QInputDialog.getItem(self, "Select Column", "Choose column for histogram:", columns, 0, False)
            if ok_col:
                if discretized_columns:
                    # Optional selection for discretization
                    discretize_column, ok_discretize = QInputDialog.getItem(
                        self, "Select Discretized Column", "Choose discretized column (optional):", discretized_columns, 0, True)
                    if ok_discretize:
                        plt.figure(figsize=(8, 6))
                        self.parent.full_data.groupby(discretize_column)[column].plot.hist(alpha=0.5, legend=True)
                        plt.title(f'Histogram of {column} grouped by {discretize_column}')
                        plt.xlabel(column)
                        plt.show()
                    else:
                        # Standard histogram without discretization
                        plt.figure(figsize=(8, 6))
                        self.parent.full_data[column].plot.hist(bins=20, alpha=0.7)
                        plt.title(f'Histogram of {column}')
                        plt.xlabel(column)
                        plt.ylabel('Frequency')
                        plt.show()
                else:
                    # No discretized columns available; plot a simple histogram
                    plt.figure(figsize=(8, 6))
                    self.parent.full_data[column].plot.hist(bins=20, alpha=0.7)
                    plt.title(f'Histogram of {column}')
                    plt.xlabel(column)
                    plt.ylabel('Frequency')
                    plt.show()
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")
    # Define the new method for heatmap plotting

    def plot_barplot(self):
        """
        Plot bar plots for categorical or interval data.
        Allows selection of multiple columns for simultaneous display.
        """
        if self.parent.full_data is not None:
            # Get columns that are categorical or interval
            categorical_or_interval_columns = [
                col for col in self.parent.full_data.columns
                if self.parent.full_data[col].dtype.name in ['category', 'object']
            ]

            if not categorical_or_interval_columns:
                QMessageBox.warning(
                    self, "No Categorical or Interval Data",
                    "No categorical or interval data found in the dataset."
                )
                return

            # Select columns for plotting
            selected_columns, ok = QInputDialog.getItem(
                self,
                "Select Columns",
                "Choose one or more columns (separate by commas):",
                categorical_or_interval_columns,
                editable=True
            )
            if ok and selected_columns:
                selected_columns = [col.strip() for col in selected_columns.split(",")]

                # Ensure selected columns are in the dataset
                valid_columns = [
                    col for col in selected_columns if col in categorical_or_interval_columns
                ]
                if not valid_columns:
                    QMessageBox.warning(self, "Invalid Columns", "None of the selected columns are valid.")
                    return

                # Plot bar plots for each selected column
                plt.figure(figsize=(10, 6))
                for i, col in enumerate(valid_columns, start=1):
                    plt.subplot(len(valid_columns), 1, i)  # Create subplots
                    self.parent.full_data[col].value_counts().plot(
                        kind='bar', title=f"Frequency of {col}", alpha=0.7
                    )
                    plt.xlabel("Categories")
                    plt.ylabel("Frequency")
                    plt.tight_layout()

                plt.show()
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")
    def plot_heatmap(self):
        if self.parent.full_data is not None:
            # Select numeric columns for correlation calculation
            excluded_columns = ['latitude', 'longitude', 'time', 'geometry']
            numeric_data = self.parent.full_data.select_dtypes(include=['float64', 'int64'])
            
            # Drop excluded columns if present
            numeric_data = numeric_data.drop(columns=[col for col in excluded_columns if col in numeric_data.columns], errors='ignore')

            if numeric_data.empty:
                QMessageBox.warning(self, "No Numeric Columns", "No numeric columns available for correlation heatmap after excluding specified columns.")
                return
            
            # Calculate the correlation matrix
            correlation_matrix = numeric_data.corr()
            plt.figure(figsize=(10, 8))
            
            # Plot the heatmap
            heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Correlation Heatmap (Excluding Latitude, Longitude, Time, Geometry)")
            plt.show()
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")
    def plot_discretized_histogram(self):
        """
        Visualizes histograms for discretized data, taking into account interval lengths.
        """
        if self.parent.full_data is not None:
            # Identify discretized columns (likely of type 'object' with intervals)
            discretized_columns = [
                col for col in self.parent.full_data.columns
                if self.parent.full_data[col].dtype == 'object' and
                self.parent.full_data[col].str.contains(r'\[|\(').any()
            ]

            if not discretized_columns:
                QMessageBox.warning(self, "No Discretized Columns", "No discretized columns found in the dataset.")
                return

            # Prompt user to select a column
            column, ok = QInputDialog.getItem(
                self, "Select Column", "Choose discretized column for histogram:", discretized_columns, 0, False
            )

            if ok and column:
                # Extract the interval counts
                interval_counts = self.parent.full_data[column].value_counts().sort_index()

                # Prepare the intervals for plotting
                intervals = interval_counts.index
                counts = interval_counts.values

                # Parse intervals for their numerical bounds
                bounds = [tuple(map(float, interval.strip("()[]").split(", "))) for interval in intervals]
                lower_bounds = [bound[0] for bound in bounds]
                upper_bounds = [bound[1] for bound in bounds]
                widths = [upper - lower for lower, upper in bounds]

                # Create the histogram using actual interval widths
                plt.figure(figsize=(10, 6))
                plt.bar(
                    lower_bounds, counts, width=widths, align='edge', alpha=0.7, edgecolor='black'
                )
                plt.xticks(
                    lower_bounds, intervals, rotation=90
                )
                plt.xlabel(f"Intervals ({column})")
                plt.ylabel("Frequency")
                plt.title(f"Histogram for {column} (Intervals with Varying Lengths)")
                plt.tight_layout()
                plt.show()

        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")





    def plot_parallel_coordinates(self):
        if self.parent.full_data is not None:
            data = self.parent.full_data.copy()

            # Convert discretized columns to midpoints
            for col in data.columns:
                if data[col].dtype == 'object' and data[col].str.contains(r'\[|\(').any():
                    data[col] = convert_intervals_to_midpoints(data[col])

            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if not numeric_columns:
                QMessageBox.warning(self, "No Numeric Columns", "No numeric or discretized columns available for parallel coordinates.")
                return

            # Add temporary grouping column
            data['group'] = data[numeric_columns[0]]

            plt.figure(figsize=(15, 8))
            from pandas.plotting import parallel_coordinates
            parallel_coordinates(data, 'group', colormap='viridis', alpha=0.7)
            plt.title("Parallel Coordinates Plot")
            plt.tight_layout()
            plt.show()
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def plot_pairwise_scatter(self):
        if self.parent.full_data is not None:
            data = self.parent.full_data.copy()

            # Convert discretized columns to midpoints
            for col in data.columns:
                if data[col].dtype == 'object' and data[col].str.contains(r'\[|\(').any():
                    data[col] = convert_intervals_to_midpoints(data[col])

            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if len(numeric_columns) < 2:
                QMessageBox.warning(self, "Insufficient Columns", "Need at least two numeric or discretized columns for scatterplot matrix.")
                return

            import seaborn as sns
            sns.pairplot(data[numeric_columns], diag_kind="kde", corner=True, plot_kws={'alpha': 0.5})
            plt.suptitle("Pairwise Scatterplot Matrix", y=1.02)
            plt.show()
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")
    
    def plot_bubble_chart(self):
        if self.parent.full_data is not None:
            data = self.parent.full_data.copy()

            # Convert discretized columns to midpoints
            for col in data.columns:
                if data[col].dtype == 'object' and data[col].str.contains(r'\[|\(').any():
                    data[col] = convert_intervals_to_midpoints(data[col])

            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if len(numeric_columns) < 3:
                QMessageBox.warning(self, "Insufficient Columns", "Need at least three numeric or discretized columns for bubble chart.")
                return

            x_column, ok_x = QInputDialog.getItem(self, "Select X Column", "Choose X column:", numeric_columns, 0, False)
            if not ok_x:
                return

            y_column, ok_y = QInputDialog.getItem(self, "Select Y Column", "Choose Y column:", numeric_columns, 1, False)
            if not ok_y:
                return

            size_column, ok_size = QInputDialog.getItem(self, "Select Size Column", "Choose Size column:", numeric_columns, 2, False)
            if not ok_size:
                return

            plt.figure(figsize=(10, 8))
            plt.scatter(data[x_column], data[y_column], 
                        s=data[size_column] * 10, alpha=0.6, cmap='coolwarm', c=data[size_column])
            plt.colorbar(label=size_column)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f"Bubble Chart: {x_column} vs {y_column} (Size: {size_column})")
            plt.show()
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def plot_heatmap(self):
        if self.parent.full_data is not None:
            data = self.parent.full_data.copy()

            # Convert discretized columns to midpoints
            for col in data.columns:
                if data[col].dtype == 'object' and data[col].str.contains(r'\[|\(').any():
                    data[col] = convert_intervals_to_midpoints(data[col])

            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if not numeric_columns:
                QMessageBox.warning(self, "No Numeric Columns", "No numeric or discretized columns available for heatmap.")
                return

            correlation_matrix = data[numeric_columns].corr()
            plt.figure(figsize=(10, 8))
            import seaborn as sns
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Correlation Heatmap")
            plt.show()
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")
