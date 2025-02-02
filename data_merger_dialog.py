from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QMessageBox, QFileDialog
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely import wkt
import ast


class DataMergerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Data Merger")
        self.setGeometry(200, 200, 400, 250)  # Adjust height to fit the new button

        # Layout for the data merger options
        layout = QVBoxLayout()

        # Button to merge data
        self.merge_button = QPushButton("Select CSV to Merge")
        self.merge_button.clicked.connect(self.merge_data)
        layout.addWidget(self.merge_button)

        # Button for reducing climate data by soil polygons
        self.reduce_button = QPushButton("Merge Climate Data With Soil Data")
        self.reduce_button.clicked.connect(self.reduce_by_soil_polygons)
        layout.addWidget(self.reduce_button)

        # Button for merging rows by geometry
        self.merge_geometry_button = QPushButton("Reduce Data by similar Geometry")
        self.merge_geometry_button.clicked.connect(self.activate_merge_rows_by_geometry)
        layout.addWidget(self.merge_geometry_button)

        # Set dialog layout
        self.setLayout(layout)

    def merge_data(self):
        if self.parent.full_data is not None:
            # Open a file dialog to select the second CSV file
            file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File to Merge", "", "CSV Files (*.csv);;All Files (*)")
            if not file_path:
                return

            # Load the second CSV file
            try:
                secondary_data = pd.read_csv(file_path)
            except Exception as e:
                QMessageBox.warning(self, "File Error", f"Could not load CSV file: {e}")
                return

            # Check if both files have 'latitude' and 'longitude' columns
            required_columns = {'latitude', 'longitude'}
            if not required_columns.issubset(self.parent.full_data.columns) or not required_columns.issubset(secondary_data.columns):
                QMessageBox.warning(self, "Missing Columns", "Both files must have 'latitude' and 'longitude' columns for merging.")
                return

            # Perform the merge on 'latitude' and 'longitude' columns
            merged_data = pd.merge(self.parent.full_data, secondary_data, on=['latitude', 'longitude'], how='inner')

            # Update the main data with merged data
            self.parent.full_data = merged_data
            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()

            QMessageBox.information(self, "Merge Complete", "Data has been successfully merged and displayed.")
        else:
            QMessageBox.warning(self, "No Data", "Please import a primary dataset first.")

    def reduce_by_soil_polygons(self):
        # Prompt user to select soil and climate data files
        soil_data_path, _ = QFileDialog.getOpenFileName(self, "Select Soil Data File", "", "CSV Files (*.csv);;All Files (*)")
        if not soil_data_path:
            QMessageBox.warning(self, "File Selection Error", "Please select a valid soil data file.")
            return

        climate_data_path, _ = QFileDialog.getOpenFileName(self, "Select Climate Data File", "", "CSV Files (*.csv);;All Files (*)")
        if not climate_data_path:
            QMessageBox.warning(self, "File Selection Error", "Please select a valid climate data file.")
            return

        # Perform the reduction
        try:
            reduced_data = self.perform_reduction(soil_data_path, climate_data_path)

            # Update the parent application's full_data and refresh the display
            self.parent.full_data = reduced_data
            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()

            QMessageBox.information(self, "Reduction Complete", "Climate data has been reduced by soil polygons.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def perform_reduction(self, soil_data_path, climate_data_path):
        try:
            # Load soil data from CSV
            soil_data = pd.read_csv(soil_data_path)
            if 'geometry' not in soil_data.columns:
                raise ValueError("Soil data must have a 'geometry' column.")

            # Convert 'geometry' column to Polygon objects
            soil_data['geometry'] = soil_data['geometry'].apply(wkt.loads)

            # Convert soil data to GeoDataFrame
            soil_gdf = gpd.GeoDataFrame(soil_data, geometry='geometry')
            soil_gdf = soil_gdf.set_crs("EPSG:4326")  # Assuming WGS84 (adjust if needed)

            # Load climate data as DataFrame
            climate_data = pd.read_csv(climate_data_path)

            # Validate climate data has 'longitude' and 'latitude' columns
            if 'longitude' not in climate_data.columns or 'latitude' not in climate_data.columns:
                raise ValueError("Climate data must contain 'longitude' and 'latitude' columns.")

            # Convert climate data to GeoDataFrame with Point geometry
            climate_data['geometry'] = climate_data.apply(
                lambda row: Point(row['longitude'], row['latitude']), axis=1
            )
            climate_gdf = gpd.GeoDataFrame(climate_data, geometry='geometry')
            climate_gdf = climate_gdf.set_crs("EPSG:4326")  # Assuming WGS84 (adjust if needed)

            # Spatial join: Map climate data points to soil polygons
            climate_with_soil = gpd.sjoin(climate_gdf, soil_gdf, how='inner', predicate='within')

            # Group climate data by soil polygon and calculate the mean for each group
            climate_columns = [col for col in climate_data.columns if col not in ['latitude', 'longitude', 'geometry']]
            aggregated_climate = climate_with_soil.groupby('index_right')[climate_columns].mean().reset_index()

            # Retain unique soil data and merge with aggregated climate data
            soil_gdf['polygon_id'] = soil_gdf.index  # Add a unique polygon ID
            reduced_data = pd.merge(soil_gdf, aggregated_climate, left_on='polygon_id', right_on='index_right', how='left')

            # Drop unnecessary columns and return reduced dataset
            reduced_data = reduced_data.drop(columns=['polygon_id', 'index_right'])
            return reduced_data

        except Exception as e:
            raise RuntimeError(f"An error occurred while processing: {str(e)}")

    def activate_merge_rows_by_geometry(self):
        """
        Activate the `merge_rows_by_geometry` function and update the dataset.
        """
        if self.parent.full_data is not None:
            try:
                # Call the merge_rows_by_geometry function
                updated_data = self.merge_rows_by_geometry(self.parent.full_data)

                # Update the parent dataset
                self.parent.full_data = updated_data
                self.parent.current_page = 0
                self.parent.update_total_pages()
                self.parent.display_data()

                QMessageBox.information(self, "Merge Complete", "Rows with the same geometry have been merged successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    @staticmethod
    def merge_rows_by_geometry(data):
        """
        Merge rows with the same geometry, calculating averages for numeric columns
        and combining latitude and longitude into a polygon, line, or point.

        Parameters:
        - data: DataFrame with 'geometry', 'latitude', and 'longitude' columns.

        Returns:
        - A DataFrame with merged rows based on the geometry column.
        """
        from shapely.geometry import LineString

        # Ensure latitude, longitude, and geometry columns exist
        if 'latitude' not in data.columns or 'longitude' not in data.columns or 'geometry' not in data.columns:
            raise ValueError("Dataset must contain 'latitude', 'longitude', and 'geometry' columns.")

        # Group by geometry
        grouped = data.groupby('geometry')

        # Initialize a list to store results
        merged_data = []

        for geom, group in grouped:
            # Calculate averages for numeric columns, excluding latitude and longitude
            numeric_averages = group.select_dtypes(include=['float64', 'int64']).mean()

            # Combine latitude and longitude into a polygon, line, or point
            points = [Point(row['longitude'], row['latitude']) for _, row in group.iterrows()]
            
            if len(points) > 2:
                # Create a polygon if there are at least three unique points
                geometry = Polygon([[point.x, point.y] for point in points])
            elif len(points) == 2:
                # Create a line if there are two points
                geometry = LineString([[point.x, point.y] for point in points])
            elif len(points) == 1:
                # Use the single point directly
                geometry = points[0]
            else:
                # Skip rows with no valid points
                continue

            # Create a new row with averaged values and the new geometry
            merged_row = numeric_averages.to_dict()
            merged_row['geometry'] = geometry

            # Append the merged row to the result
            merged_data.append(merged_row)

        # Convert the list of merged rows into a DataFrame
        merged_df = pd.DataFrame(merged_data)

        return merged_df
