import sys
import pandas as pd
import tempfile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QTableWidget,
                             QTableWidgetItem, QLabel, QHBoxLayout, QInputDialog, QMessageBox, QSplitter, QComboBox, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from data_manipulation_dialog import DataManipulationDialog
from data_cleaning_dialog import DataCleaningDialog
from data_normalization_dialog import DataNormalizationDialog
from data_aggregation_dialog import DataAggregationDialog
from data_discretization_dialog import DataDiscretizationDialog
from data_visualization_dialog import DataVisualizationDialog
from data_merger_dialog import DataMergerDialog
from algorithms_decision_tree import DecisionTreeAlgorithm
from algorithms_random_forest import RandomForestAlgorithm
from algorithms_clarans import ClaransAlgorithm
from algorithms_dbscan import DbScanAlgorithm
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLineEdit
import numpy as np

class PredictInputDialog(QDialog):
    def __init__(self, feature_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter Feature Values for Prediction")
        self.feature_names = feature_names
        self.feature_inputs = {}

        # Create a grid layout for parallel inputs
        layout = QGridLayout()

        # Create input fields for each feature
        for i, feature in enumerate(feature_names):
            label = QLabel(feature)
            input_field = QLineEdit()
            input_field.setPlaceholderText(f"Enter value for {feature}")
            self.feature_inputs[feature] = input_field

            # Add label and input field to the grid
            row = i // 2  # Two inputs per row
            col = (i % 2) * 2  # Alternate between column 0 and 2
            layout.addWidget(label, row, col)
            layout.addWidget(input_field, row, col + 1)

        # Add OK and Cancel buttons at the bottom
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        # Add button layout to the grid
        layout.addLayout(button_layout, (len(feature_names) + 1) // 2, 0, 1, 4)

        self.setLayout(layout)

    def get_inputs(self):
        """Retrieve and validate user inputs."""
        try:
            # Convert all inputs to floats
            inputs = {feature: float(field.text()) for feature, field in self.feature_inputs.items()}
            return list(inputs.values())  # Return as a list for model prediction
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please ensure all fields are filled with numeric values.")
            return None


class DbScanParametersDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dbscan Parameters")

        self.layout = QVBoxLayout()

        # Champ pour eps
        self.eps_label = QLabel("Enter the eps:")
        self.eps_input = QLineEdit()
        self.layout.addWidget(self.eps_label)
        self.layout.addWidget(self.eps_input)

        # Champ pour min_pts
        self.min_pts_label = QLabel("Enter the min_pts:")
        self.min_pts_input = QLineEdit()
        self.layout.addWidget(self.min_pts_label)
        self.layout.addWidget(self.min_pts_input)

        # Liste déroulante pour le choix des options
        self.option_label = QLabel("Choose DBSCAN Mode:")
        self.option_dropdown = QComboBox()
        self.option_dropdown.addItems(["Without PCA", "With PCA"])
        self.layout.addWidget(self.option_label)
        self.layout.addWidget(self.option_dropdown)

        # Bouton pour valider
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

        self.setLayout(self.layout)

    def get_inputs(self):
        return (
            float(self.eps_input.text()),
            float(self.min_pts_input.text()),
            self.option_dropdown.currentText(),  # Retourne l'option sélectionnée
        )
    

class ClaransParametersDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clarans Parameters")

        self.layout = QVBoxLayout()

        # Champ pour num_clusters
        self.num_clusters_label = QLabel("Enter the num_clusters:")
        self.num_clusters_input = QLineEdit()
        self.layout.addWidget(self.num_clusters_label)
        self.layout.addWidget(self.num_clusters_input)

        # Champ pour max_neighbors
        self.max_neighbors_label = QLabel("Enter the max_neighbors:")
        self.max_neighbors_input = QLineEdit()
        self.layout.addWidget(self.max_neighbors_label)
        self.layout.addWidget(self.max_neighbors_input)

        # Champ pour num_local
        self.num_local_label = QLabel("Enter the num_local:")
        self.num_local_input = QLineEdit()
        self.layout.addWidget(self.num_local_label)
        self.layout.addWidget(self.num_local_input)

        # Liste déroulante pour le choix des options
        self.option_label = QLabel("Choose CLARANS Mode:")
        self.option_dropdown = QComboBox()
        self.option_dropdown.addItems(["Without PCA", "With PCA"])
        self.layout.addWidget(self.option_label)
        self.layout.addWidget(self.option_dropdown)

        # Bouton pour valider
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

        self.setLayout(self.layout)

    def get_inputs(self):
        return (
            int(self.num_clusters_input.text()),
            int(self.max_neighbors_input.text()),
            int(self.num_local_input.text()),
            self.option_dropdown.currentText(),  # Retourne l'option sélectionnée
        )
    
class RandomForestParametersDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Random Forest Parameters")

        self.layout = QVBoxLayout()

        # Champ pour la colonne cible
        self.target_label = QLabel("Enter the target column:")
        self.target_input = QLineEdit()
        self.layout.addWidget(self.target_label)
        self.layout.addWidget(self.target_input)

        # Champ pour min_samples_split
        self.min_samples_split_label = QLabel("Enter the min_samples_split:")
        self.min_samples_split_input = QLineEdit()
        self.layout.addWidget(self.min_samples_split_label)
        self.layout.addWidget(self.min_samples_split_input)

        # Champ pour max_depth
        self.max_depth_label = QLabel("Enter the max_depth:")
        self.max_depth_input = QLineEdit()
        self.layout.addWidget(self.max_depth_label)
        self.layout.addWidget(self.max_depth_input)

        # Champ pour n_estimators
        self.n_estimators_label = QLabel("Enter the n_estimators:")
        self.n_estimators_input = QLineEdit()
        self.layout.addWidget(self.n_estimators_label)
        self.layout.addWidget(self.n_estimators_input)

        # Bouton pour valider
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

        self.setLayout(self.layout)

    def get_inputs(self):
        return (
            self.target_input.text(),
            int(self.min_samples_split_input.text()),
            int(self.max_depth_input.text()),
            int(self.n_estimators_input.text()),
        )


class DecisionTreeParametersDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Decision Trees Parameters")

        self.layout = QVBoxLayout()

        # Champ pour la colonne cible
        self.target_label = QLabel("Enter the target column:")
        self.target_input = QLineEdit()
        self.layout.addWidget(self.target_label)
        self.layout.addWidget(self.target_input)

        # Champ pour min_samples_split
        self.min_samples_split_label = QLabel("Enter the min_samples_split:")
        self.min_samples_split_input = QLineEdit()
        self.layout.addWidget(self.min_samples_split_label)
        self.layout.addWidget(self.min_samples_split_input)

        # Champ pour max_depth
        self.max_depth_label = QLabel("Enter the max_depth:")
        self.max_depth_input = QLineEdit()
        self.layout.addWidget(self.max_depth_label)
        self.layout.addWidget(self.max_depth_input)

        # Bouton pour valider
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

        self.setLayout(self.layout)

    def get_inputs(self):
        return (
            self.target_input.text(),
            int(self.min_samples_split_input.text()),
            int(self.max_depth_input.text()),
        )

class CsvViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Mining Utility")
        self.setGeometry(100, 100, 1000, 600)
        self.setWindowIcon(QIcon("Cloud.jpeg"))  # Remplacez "Cloud.jpeg" par le chemin de votre logo

        # Initialisation des variables pour la pagination
        self.full_data = None
        self.chunk_size = 1000  # Taille des blocs de données à afficher par page
        self.current_page = 0   # Page actuelle
        self.total_pages = 0    # Nombre total de pages

        # Initialisation de l'interface utilisateur
         # Temporary file to store the last state of the dataset
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        self.init_ui()
    def save_current_state(self):
        """Saves the current state of the dataset to a temporary file."""
        if self.full_data is not None:
            self.full_data.to_csv(self.temp_file.name, index=False)
    def restore_last_state(self):
        """Restores the last state of the dataset from the temporary file."""
        try:
            self.full_data = pd.read_csv(self.temp_file.name)
            self.current_page = 0
            self.display_data()
            QMessageBox.information(self, "Undo Successful", "The last modification has been undone.")
        except Exception as e:
            QMessageBox.warning(self, "Restore Failed", f"Failed to restore the dataset: {e}")
    def init_ui(self):
        # Configuration de la mise en page principale
        main_layout = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)

        # Barre latérale gauche contenant les boutons
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout()
        buttons = [
            ("Data Manipulation", self.open_data_manipulation),
            ("Data Cleaning", self.open_data_cleaning),
            ("Data Normalization", self.open_data_normalization),
            ("Data Aggregation", self.open_data_aggregation),
            ("Data Merger", self.open_data_merger),
            ("Data Discretization", self.open_data_discretization),
            ("Data Visualization", self.open_data_visualization),
            ("Copy Selection", self.copy_selection),
            ("Undo Last Modification", self.restore_last_state),
        ]
        for text, func in buttons:
            button = QPushButton(text)
            button.clicked.connect(func)  # Connecte chaque bouton à sa fonction respective
            button.setFixedHeight(40)
            sidebar_layout.addWidget(button)
        sidebar.setLayout(sidebar_layout)
        sidebar.setFixedWidth(200)  # Largeur fixe de la barre latérale
        sidebar_layout.addStretch()

        # Barre latérale droite pour les algorithmes
        algo_sidebar = QWidget()
        algo_layout = QVBoxLayout(algo_sidebar)
        algo_buttons = [
            ("Decision Trees", self.run_decision_trees),
            ("Random Forest", self.run_random_forest),
            ("CLARANS", self.run_clarans),
            ("DBSCAN", self.run_dbscan),
        ]
        for text, func in algo_buttons:
            button = QPushButton(text)
            button.clicked.connect(func)  # Connecte chaque bouton à sa fonction respective
            button.setFixedHeight(40)
            algo_layout.addWidget(button)
        algo_sidebar.setLayout(algo_layout)
        algo_sidebar.setFixedWidth(200)
        algo_layout.addStretch()

        # Zone d'affichage des données
        data_display_area = QWidget()
        data_layout = QVBoxLayout(data_display_area)
        self.table = QTableWidget()  # Tableau pour afficher les données
        self.table.setSelectionBehavior(QTableWidget.SelectItems)  # Sélection cellule par cellule
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)  # Sélection de plusieurs cellules
        data_layout.addWidget(self.table)

        # Contrôles de pagination
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")  # Bouton page précédente
        self.prev_button.clicked.connect(self.load_previous_page)
        self.prev_button.setEnabled(False)  # Désactivé par défaut
        nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next")  # Bouton page suivante
        self.next_button.clicked.connect(self.load_next_page)
        self.next_button.setEnabled(False)  # Désactivé par défaut
        nav_layout.addWidget(self.next_button)

        # Bouton pour aller à une page spécifique
        self.jump_button = QPushButton("Go to Page")
        self.jump_button.clicked.connect(self.go_to_page)
        nav_layout.addWidget(self.jump_button)

        self.page_label = QLabel("Page: 0/0")  # Indicateur de pagination
        nav_layout.addWidget(self.page_label)
        data_layout.addLayout(nav_layout)

        # Configuration du splitter et du widget central
        splitter.addWidget(sidebar)
        splitter.addWidget(data_display_area)
        splitter.addWidget(algo_sidebar)
        main_layout.addWidget(splitter)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.apply_styles()  # Applique les styles à l'interface

    def run_decision_trees(self):
        """Runs Decision Trees algorithm."""
        if self.full_data is not None:
            try:
                # Ouvrir la fenêtre pour saisir les paramètres
                dialog = DecisionTreeParametersDialog(self)
                if dialog.exec_() == QDialog.Accepted:
                    target_column, min_samples_split, max_depth = dialog.get_inputs()

                    # Vérifier si la colonne existe
                    if target_column not in self.full_data.columns:
                        QMessageBox.warning(self, "Invalid Column", f"The column '{target_column}' does not exist in the dataset.")
                        return

                    # Instancier et appliquer l'algorithme
                    dt_algorithm = DecisionTreeAlgorithm(self.full_data)
                    model, training_time, tree_image_path, X_test, Y_test = dt_algorithm.apply(
                        target_column=target_column,
                        min_samples_split=min_samples_split,
                        max_depth=max_depth,
                    )

                    # Afficher le temps d'entraînement dans une boîte de dialogue
                    QMessageBox.information(
                        self, 
                        "Decision Trees",
                        f"Decision Trees algorithm ran successfully.\n"
                        f"Training time: {training_time:.4f} seconds.\n"
                    )

                    # Ouvrir une fenêtre pour afficher l'image de l'arbre
                    dialog = QDialog(self)
                    dialog.setWindowTitle("Decision Tree Visualization")
                    layout = QVBoxLayout()

                    label = QLabel()
                    pixmap = QPixmap(tree_image_path)
                    label.setPixmap(pixmap)

                    layout.addWidget(label)
                    dialog.setLayout(layout)
                    dialog.exec_()

                    # Après affichage, demander l'index pour prédiction
                    index, ok = QInputDialog.getInt(
                        self,
                        "Prediction Input",
                        f"Enter an index (0 to {len(X_test)-1}) to predict:"
                    )

                    if ok:
                        if 0 <= index < len(X_test):
                            # Effectuer la prédiction
                            prediction_time, Y_pred, MSE = dt_algorithm.prediction(X_test, Y_test, model)
                            print(X_test[index])
                            QMessageBox.information(
                                self,
                                "Prediction Results",
                                f"Prediction for index {index}:\n"
                                f"value to predict: {X_test[index]}\n"
                                f"Predicted value: {Y_pred[index]}\n"
                                f"Actual value: {Y_test[index][0]}\n"
                                f"Mean Squared Error (MSE): {MSE:.4f}\n"
                                f"Prediction time: {prediction_time:.4f} seconds.\n"
                            )

                    # Open input dialog for custom prediction
                    feature_names = [
                        "latitude", "longitude", "sand % topsoil", "silt % topsoil",
                        "clay % topsoil", "pH water topsoil", "OC % topsoil", "OC % subsoil",
                        "N % topsoil", "N % subsoil", "CEC topsoil", "CaCO3 % topsoil",
                        "C/N topsoil", "C/N subsoil", "PSurfFall", "QairFall", "TairFall",
                        "TairSummer", "WindFall", "WindSpring", "WindWinter"
                    ]

                    # Drop target_column from feature_names if it exists
                    if target_column in feature_names:
                        feature_names.remove(target_column)

                    input_dialog = PredictInputDialog(feature_names, self)

                    if input_dialog.exec_() == QDialog.Accepted:
                        user_inputs = input_dialog.get_inputs()
                        if user_inputs is None:
                            return  # Invalid inputs

                        # Prepare input for model prediction
                        X_new = user_inputs
                        prediction, X_to_predict = dt_algorithm.predict_value(X_new, model, feature_names)
                        
                        QMessageBox.information(
                            self,
                            "Prediction Results",
                            f"Prediction for input values {X_to_predict}:\n"
                            f"Predicted value: {prediction[0]}"
                        )
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
        else:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")

    def run_random_forest(self):
        """Runs Random Forest algorithm."""
        if self.full_data is not None:
            try:
                # Ouvrir la fenêtre pour saisir les paramètres
                dialog = RandomForestParametersDialog(self)
                if dialog.exec_() == QDialog.Accepted:
                    target_column, min_samples_split, max_depth, n_estimators = dialog.get_inputs()

                    # Vérifier si la colonne existe
                    if target_column not in self.full_data.columns:
                        QMessageBox.warning(self, "Invalid Column", f"The column '{target_column}' does not exist in the dataset.")
                        return
                    
                    # Instancier et appliquer l'algorithme
                    rf_algorithm = RandomForestAlgorithm(self.full_data)
                    model, training_time, X_test, Y_test = rf_algorithm.apply(
                        target_column=target_column,
                        min_samples_split=min_samples_split,
                        max_depth=max_depth,
                        n_estimators=n_estimators,
                    )

                    # Afficher le temps d'entraînement dans une boîte de dialogue
                    QMessageBox.information(
                        self, 
                        "Random Forest",
                        f"Random Forest algorithm ran successfully.\n"
                        f"Training time: {training_time:.4f} seconds.\n"
                    )

                    # Ouvrir une fenêtre pour afficher l'image de l'arbre
                    # demander l'index pour l'arbre à afficher
                    index1, ok = QInputDialog.getInt(
                        self,
                        "tree index Input",
                        f"Enter an index (0 to {n_estimators}):"
                    )

                    if ok:
                        if 0 <= index1 < len(X_test):

                            tree_image_path = rf_algorithm.draw_tree(model, index1)
                            dialog = QDialog(self)
                            dialog.setWindowTitle("Decision Tree Visualization")
                            layout = QVBoxLayout()

                            label = QLabel()
                            pixmap = QPixmap(tree_image_path)
                            label.setPixmap(pixmap)

                            layout.addWidget(label)
                            dialog.setLayout(layout)
                            dialog.exec_()

                    # Après affichage, demander l'index pour prédiction
                    index, ok = QInputDialog.getInt(
                        self,
                        "Prediction Input",
                        f"Enter an index (0 to {len(X_test)-1}) to predict:"
                    )

                    if ok:
                        if 0 <= index < len(X_test):
                            # Effectuer la prédiction
                            prediction_time, Y_pred, MSE = rf_algorithm.prediction(X_test, Y_test, model)
                            
                            QMessageBox.information(
                                self,
                                "Prediction Results",
                                f"Prediction for index {index}:\n"
                                f"value to predict: {X_test[index]}\n"
                                f"Predicted value: {Y_pred[index]}\n"
                                f"Actual value: {Y_test.iloc[index]}\n"
                                f"Mean Squared Error (MSE): {MSE:.4f}\n"
                                f"Prediction time: {prediction_time:.4f} seconds.\n"
                            )

                    # Open input dialog for custom prediction
                    feature_names = [
                        "latitude", "longitude", "sand % topsoil", "silt % topsoil",
                        "clay % topsoil", "pH water topsoil", "OC % topsoil", "OC % subsoil",
                        "N % topsoil", "N % subsoil", "CEC topsoil", "CaCO3 % topsoil",
                        "C/N topsoil", "C/N subsoil", "PSurfFall", "QairFall", "TairFall",
                        "TairSummer", "WindFall", "WindSpring", "WindWinter"
                    ]

                    # Drop target_column from feature_names if it exists
                    if target_column in feature_names:
                        feature_names.remove(target_column)

                    input_dialog = PredictInputDialog(feature_names, self)

                    if input_dialog.exec_() == QDialog.Accepted:
                        user_inputs = input_dialog.get_inputs()
                        if user_inputs is None:
                            return  # Invalid inputs

                        # Prepare input for model prediction
                        X_new = user_inputs
                        prediction, X_to_predict = rf_algorithm.predict_value(X_new, model, feature_names)
                        
                        QMessageBox.information(
                            self,
                            "Prediction Results",
                            f"Prediction for input values {X_to_predict}:\n"
                            f"Predicted value: {prediction[0]}"
                        )
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
        else:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")

    def run_clarans(self):
        """Runs CLARANS clustering algorithm."""
        if self.full_data is not None:
            try:
                # Ouvrir la fenêtre pour saisir les paramètres
                dialog = ClaransParametersDialog(self)
                if dialog.exec_() == QDialog.Accepted:
                    num_clusters, max_neighbors, num_local, mode = dialog.get_inputs()

                    # Instancier et appliquer l'algorithme
                    clarans_algorithm = ClaransAlgorithm(self.full_data)
                    clusters, medoids, training_time, data_for_clustering = clarans_algorithm.apply(
                        num_clusters=num_clusters,
                        max_neighbors=max_neighbors, 
                        num_local=num_local,
                        mode=mode,
                    )
                    
                    mse_values, wc_sse, silhouette = clarans_algorithm.evaluate_clarans(clusters, medoids, data_for_clustering)

                    # MSE global (moyenne des MSE des clusters)
                    global_mse = np.mean(mse_values)
                    print(f"MSE global : {global_mse:.4f}")
                    print(f"WC-SSE : {wc_sse:.4f}")
                    print(f"silhouette_score : {silhouette:.4f}")

                    # Afficher le temps d'entraînement dans une boîte de dialogue
                    QMessageBox.information(
                        self, 
                        "Clarans",
                        f"Clarans algorithm ran successfully.\n"
                        f"Training time: {training_time:.4f} seconds.\n"
                        f"MSE global : {global_mse:.4f}.\n"  
                        f"WC-SSE : {wc_sse:.4f}.\n"
                        f"silhouette_score : {silhouette:.4f}"
                    )
                    
                    #display clusters
                    if mode == "Without PCA":
                        clarans_algorithm.display_clusters(clusters, medoids, data_for_clustering, self.full_data)
                    elif mode == "With PCA":
                        clarans_algorithm.display_clusters_PCA(clusters, medoids, data_for_clustering, self.full_data)
                     
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
        else:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")

    def run_dbscan(self):
        """Runs DBSCAN clustering algorithm."""
        if self.full_data is not None:
            try:
                # Ouvrir la fenêtre pour saisir les paramètres
                dialog = DbScanParametersDialog(self)
                if dialog.exec_() == QDialog.Accepted:
                    min_pts, eps, mode = dialog.get_inputs()

                    # Instancier et appliquer l'algorithme
                    dbscan_algorithm = DbScanAlgorithm(self.full_data)  
                    num_clusters, num_noise, training_time, d, cluster_labels = dbscan_algorithm.apply_Without_PCA(
                        min_pts, 
                        eps,
                        mode,
                    )
                    mse_values, wc_sse, silhouette = dbscan_algorithm.evaluate_dbscan(cluster_labels, d)

                    # MSE global (moyenne des MSE des clusters)
                    global_mse = np.mean(mse_values)
                    print(f"MSE global : {global_mse:.4f}")
                    print(f"WC-SSE : {wc_sse:.4f}")
                    print(f"silhouette_score : {silhouette:.4f}")

                    # Afficher le temps d'entraînement dans une boîte de dialogue
                    QMessageBox.information(
                        self, 
                        "Dbscan",
                        f"Dbscan algorithm ran successfully.\n"
                        f"Training time: {training_time:.4f} seconds.\n"
                        f"Number of clusters: {num_clusters}.\n"
                        f"Number of noise points: {num_noise}.\n"
                        f"MSE global : {global_mse:.4f}.\n"  
                        f"WC-SSE : {wc_sse:.4f}.\n"
                        f"silhouette_score : {silhouette:.4f}"
                    )

                    dbscan_algorithm.display_clusters(d, cluster_labels) 
                     
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
        else:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")


    def copy_selection(self):
        """Copie les cellules sélectionnées dans le presse-papiers."""
        selected_ranges = self.table.selectedRanges()
        if not selected_ranges:
            QMessageBox.warning(self, "No Selection", "Veuillez sélectionner une section de données à copier.")
            return

        selected_data = ""
        for range_ in selected_ranges:
            for row in range(range_.topRow(), range_.bottomRow() + 1):
                row_data = []
                for col in range(range_.leftColumn(), range_.rightColumn() + 1):
                    item = self.table.item(row, col)
                    row_data.append(item.text() if item else "")
                selected_data += "\t".join(row_data) + "\n"

        # Copie dans le presse-papiers
        clipboard = QApplication.clipboard()
        clipboard.setText(selected_data)
        QMessageBox.information(self, "Copy Complete", "Les données sélectionnées ont été copiées.")

    def apply_styles(self):
        """Applique un style personnalisé à l'interface utilisateur."""
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                color: white;
            }
            QPushButton {
                background-color: #2a1e33;
                color: #ffffff;
                border-radius: 8px;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #3b2a4e;
            }
            QTableWidget {
                background-color: #1a1a1a;
                color: #ffffff;
                font-size: 13px;
                gridline-color: #333333;
            }
            QLabel {
                font-size: 14px;
                color: #ffffff;
            }
            QMainWindow {
                border-radius: 15px;
            }
        """)
        font = QFont("Verdana", 10, QFont.Bold)
        self.setFont(font)

    def go_to_page(self):
        """Permet d'aller à une page spécifique."""
        if self.full_data is not None:
            page_num, ok = QInputDialog.getInt(self, "Go to Page", f"Entrez le numéro de page (1-{self.total_pages}):", min=1, max=self.total_pages)
            if ok:
                self.current_page = page_num - 1
                self.display_data()

    def open_data_visualization(self):
        """Ouvre la fenêtre de visualisation des données."""
        dialog = DataVisualizationDialog(self)
        dialog.exec_()

    def open_data_discretization(self):
        """Ouvre la fenêtre de discrétisation des données."""
        dialog = DataDiscretizationDialog(self)
        dialog.exec_()

    def open_data_merger(self):
        """Ouvre la fenêtre de fusion des données."""
        dialog = DataMergerDialog(self)
        dialog.exec_()

    def open_data_aggregation(self):
        """Ouvre la fenêtre d'agrégation des données."""
        dialog = DataAggregationDialog(self)
        dialog.exec_()

    def open_data_manipulation(self):
        """Ouvre la fenêtre de manipulation des données."""
        dialog = DataManipulationDialog(self)
        dialog.exec_()

    def open_data_cleaning(self):
        """Ouvre la fenêtre de nettoyage des données."""
        dialog = DataCleaningDialog(self)
        dialog.exec_()

    def open_data_normalization(self):
        """Ouvre la fenêtre de normalisation des données."""
        dialog = DataNormalizationDialog(self)
        dialog.exec_()

    def update_total_pages(self):
        """Met à jour le nombre total de pages en fonction des données disponibles."""
        if self.full_data is not None:
            self.total_pages = (len(self.full_data) // self.chunk_size) + (1 if len(self.full_data) % self.chunk_size != 0 else 0)
            self.page_label.setText(f"Page: {self.current_page + 1}/{self.total_pages}")

    def display_data(self):
        """Affiche les données de la page actuelle dans le tableau."""
        if self.full_data is None:
            return

        # Définit les indices de début et de fin pour la page actuelle
        start_row = self.current_page * self.chunk_size
        end_row = min(start_row + self.chunk_size, len(self.full_data))
        chunk = self.full_data.iloc[start_row:end_row]

        # Met à jour le tableau avec les nouvelles données
        self.table.setRowCount(chunk.shape[0])
        self.table.setColumnCount(chunk.shape[1])
        self.table.setHorizontalHeaderLabels(chunk.columns)

        for i, (index, row) in enumerate(chunk.iterrows()):
            for j, value in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(str(value)))
            self.table.setVerticalHeaderItem(i, QTableWidgetItem(str(index + 1)))  # Définit l'indice des lignes

        self.update_total_pages()
        self.prev_button.setEnabled(self.current_page > 0)
        self.next_button.setEnabled(self.current_page < self.total_pages - 1)

    def load_previous_page(self):
        """Charge la page précédente."""
        if self.current_page > 0:
            self.current_page -= 1
            self.display_data()

    def load_next_page(self):
        """Charge la page suivante."""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.display_data()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = CsvViewer()
    viewer.show()
    sys.exit(app.exec_())
