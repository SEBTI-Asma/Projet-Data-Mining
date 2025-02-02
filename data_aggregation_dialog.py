import pandas as pd
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QMessageBox

class DataAggregationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Data Aggregation")
        self.setGeometry(200, 200, 400, 200)  # Définit la position et la taille de la fenêtre

        # Création de la mise en page verticale pour contenir les boutons
        layout = QVBoxLayout()

        # Bouton pour l'agrégation mensuelle
        self.monthly_button = QPushButton("Aggregate Monthly")
        self.monthly_button.clicked.connect(self.aggregate_monthly)  # Connecte le bouton à la méthode correspondante
        layout.addWidget(self.monthly_button)

        # Bouton pour l'agrégation saisonnière
        self.seasonal_button = QPushButton("Aggregate Seasonally")
        self.seasonal_button.clicked.connect(self.aggregate_seasonally)  # Connecte le bouton à la méthode correspondante
        layout.addWidget(self.seasonal_button)

        # Bouton pour la réduction des données par saison
        self.reduction_button = QPushButton("Reduce Data by Season")
        self.reduction_button.clicked.connect(self.reduce_data_by_season)  # Connecte le bouton à la méthode correspondante
        layout.addWidget(self.reduction_button)

        # Applique la mise en page à la boîte de dialogue
        self.setLayout(layout)

    def reduce_data_by_season(self):
        """Réduit les données en fonction de la saison."""
        if self.parent.full_data is not None:
            try:
                # Réorganise les données pour qu'elles soient pivotées par 'latitude', 'longitude' et 'saison'
                pivot_df = self.parent.full_data.pivot(
                    index=['latitude', 'longitude'],  # Index pour le pivot (clé unique pour chaque entrée)
                    columns='season',  # Les colonnes seront organisées par saison
                    values=['PSurf', 'Qair', 'Rainf', 'Snowf', 'Tair', 'Wind']  # Variables pour lesquelles appliquer le pivot
                )

                # Aplatissement des colonnes à plusieurs niveaux pour créer des noms de colonnes uniques
                pivot_df.columns = [f"{var}{season}" for var, season in pivot_df.columns]

                # Réinitialisation de l'index pour fusionner avec les autres données
                pivot_df.reset_index(inplace=True)

                # Mise à jour des données principales avec la version réduite
                self.parent.full_data = pivot_df
                self.parent.current_page = 0  # Réinitialise à la première page
                self.parent.update_total_pages()  # Met à jour le nombre total de pages
                self.parent.display_data()  # Actualise l'affichage des données

                QMessageBox.information(self, "Reduction Complete", "Data has been reduced by season and displayed.")
            except Exception as e:
                QMessageBox.warning(self, "Reduction Error", f"An error occurred during data reduction: {e}")
        else:
            QMessageBox.warning(self, "No Data", "Please import a dataset first.")

    def ensure_datetime(self):
        """Vérifie et convertit la colonne 'time' au format datetime."""
        if 'time' in self.parent.full_data.columns:
            self.parent.full_data['time'] = pd.to_datetime(self.parent.full_data['time'], errors='coerce')  # Conversion en datetime
            if self.parent.full_data['time'].isnull().any():  # Vérifie si des valeurs n'ont pas pu être converties
                QMessageBox.warning(self, "Date Conversion Error", "Certaines valeurs dans 'time' n'ont pas pu être converties.")
                self.parent.full_data.dropna(subset=['time'], inplace=True)  # Supprime les lignes avec des valeurs invalides
            return True
        else:
            QMessageBox.warning(self, "Missing 'time' Column", "Les données doivent contenir une colonne 'time' au format datetime.")
            return False

    def aggregate_monthly(self):
        """Agrège les données sur une base mensuelle."""
        if self.parent.full_data is not None:
            # Vérifie la présence des colonnes nécessaires
            if not {'time', 'latitude', 'longitude'}.issubset(self.parent.full_data.columns):
                QMessageBox.warning(self, "Missing Columns", "Les données doivent contenir les colonnes 'time', 'latitude', et 'longitude'.")
                return

            if not self.ensure_datetime():  # Vérifie si la colonne 'time' est au format datetime
                return

            # Extrait l'année et le mois pour le regroupement mensuel
            self.parent.full_data['year_month'] = self.parent.full_data['time'].dt.to_period('M')

            # Effectue l'agrégation par 'latitude', 'longitude', et 'year_month', en calculant la moyenne
            monthly_data = self.parent.full_data.groupby(
                ['latitude', 'longitude', 'year_month']
            ).mean(numeric_only=True).reset_index()

            # Met à jour les données principales avec les données agrégées
            self.parent.full_data = monthly_data
            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()

            QMessageBox.information(self, "Aggregation Complete", "Les données ont été agrégées mensuellement et affichées.")

    def aggregate_seasonally(self):
        """Agrège les données sur une base saisonnière."""
        if self.parent.full_data is not None:
            # Vérifie la présence des colonnes nécessaires
            if not {'time', 'latitude', 'longitude'}.issubset(self.parent.full_data.columns):
                QMessageBox.warning(self, "Missing Columns", "Les données doivent contenir les colonnes 'time', 'latitude', et 'longitude'.")
                return

            if not self.ensure_datetime():  # Vérifie si la colonne 'time' est au format datetime
                return

            # Définit les saisons en fonction des dates
            def get_season(date):
                """Détermine la saison en fonction de la date."""
                if date >= pd.Timestamp(f"{date.year}-12-21") or date < pd.Timestamp(f"{date.year}-03-20"):
                    return 'Winter'  # Hiver
                elif pd.Timestamp(f"{date.year}-03-21") <= date < pd.Timestamp(f"{date.year}-06-20"):
                    return 'Spring'  # Printemps
                elif pd.Timestamp(f"{date.year}-06-21") <= date < pd.Timestamp(f"{date.year}-09-22"):
                    return 'Summer'  # Été
                elif pd.Timestamp(f"{date.year}-09-23") <= date < pd.Timestamp(f"{date.year}-12-20"):
                    return 'Fall'  # Automne

            # Ajoute une colonne 'season' en fonction des dates
            self.parent.full_data['season'] = self.parent.full_data['time'].apply(get_season)

            # Effectue l'agrégation saisonnière par 'latitude', 'longitude', et 'season'
            seasonal_data = self.parent.full_data.groupby(
                ['latitude', 'longitude', 'season']
            ).mean(numeric_only=True).reset_index()

            # Met à jour les données principales avec les données agrégées
            self.parent.full_data = seasonal_data
            self.parent.current_page = 0
            self.parent.update_total_pages()
            self.parent.display_data()

            QMessageBox.information(self, "Aggregation Complete", "Les données ont été agrégées saisonnièrement et affichées.")
