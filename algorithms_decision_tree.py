import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from graphviz import Digraph
from sklearn.metrics import mean_squared_error

class Node(): #C'est une classe pour représenter un nœud dans l'arbre de décision.
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        ''' constructor ''' 
        # Initialise les attributs selon qu'il s'agit d'un nœud décisionnel ou d'un nœud feuille
        # for decision node
        self.feature_index = feature_index # L'indice de la caractéristique utilisée pour le split.
        self.threshold = threshold # La valeur seuil utilisée pour le split.
        self.left = left # Sous-arbres gauche
        self.right = right # Sous-arbres droit
        self.var_red = var_red #  La réduction de la variance après le split
        # Une grande valeur de var_red signifie que la division est efficace pour séparer les données en groupes
        
        # for leaf node
        self.value = value # La valeur pour les nœuds feuilles (moyenne des sorties dans la feuille).

class DecisionTreeRegressor(): # Implémente l'algorithme d'arbre de décision pour la régression.
    def __init__(self, min_samples_split=2, max_depth=4):
        ''' constructor '''
        
        # La racine de l'arbre.
        self.root = None
        
        # stopping conditions
        # si un nœud contient moins de min_samples_split échantillons, il ne sera pas divisé, et ce nœud deviendra une feuille.
        self.min_samples_split = min_samples_split # Nombre minimal d'échantillons requis pour effectuer un split.
        self.max_depth = max_depth # Profondeur maximale de l'arbre
        
    def build_tree(self, dataset, curr_depth=0):
        ''' Fonction récursive pour construire l'arbre '''
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        # si les conditions d'arrets ne sont pas verifiées elle trouve la meilleure séparation 
        # (get_best_split) et construit récursivement les sous-arbres gauche et droit.
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["var_red"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["var_red"])
        
        # Si les conditions d'arrêt sont atteintes (profondeur maximale ou échantillons insuffisants), un nœud feuille est créé.
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' cherche la meilleure division (split) '''
        
        # dictionary to store the best split
        best_split = {}
        max_var_red = -float("inf")
        # boucler sur toutes les features possibles
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    # Pour chaque combinaison, il calcule la réduction de la variance (var_red) 
                    # et choisit celle qui maximise cette valeur.
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    # update the best split if needed
                    if curr_var_red>max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' Sépare le dataset en deux parties en fonction du seuil pour une caractéristique donnée '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def variance_reduction(self, parent, l_child, r_child):
        ''' Calcule la réduction de variance obtenue après un split. '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction
    
    def calculate_leaf_value(self, Y):
        ''' Calcule la valeur du nœud feuille (moyenne des sorties) '''
        
        val = np.mean(Y)
        return val
                
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' Entraîne l'arbre en appelant build_tree. '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        
    def make_prediction(self, x, tree):
        ''' Fonction récursive pour prédire la valeur cible pour un seul échantillon '''
        
        # Si le nœud actuel est une feuille (c'est-à-dire qu'il contient une valeur prédite),
        # on retourne directement cette valeur.
        if tree.value!=None: return tree.value 
        # Récupère la valeur de la caractéristique (feature) correspondant à l'index du nœud.
        feature_val = x[tree.feature_index]
        # Vérifie si la valeur de la caractéristique est inférieure ou égale au seuil du nœud.
        # Si oui, on explore le sous-arbre gauche.
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
    def predict(self, X):
        ''' Prédit les sorties pour un ensemble d'échantillons en parcourant l'arbre '''
        
        # Applique la méthode `make_prediction` pour chaque échantillon dans l'ensemble X.
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def visualize_tree(self, tree=None, dot=None, parent_name=None, yes_name=None, no_name=None):
        ''' Fonction pour visualiser l'arbre de décision avec graphviz '''
        if not dot:
            dot = Digraph()
        
        if not tree:
            tree = self.root
        
        if tree.value is not None:
            dot.node(name=str(id(tree)), label=str(tree.value))
            if parent_name:
                dot.edge(parent_name, str(id(tree)))
        else:
            dot.node(name=str(id(tree)), label=f"X{tree.feature_index} <= {tree.threshold:.2f}")
            if parent_name:
                dot.edge(parent_name, str(id(tree)))
            
            # Create left child node
            self.visualize_tree(tree.left, dot=dot, parent_name=str(id(tree)), yes_name=str(id(tree.left)), no_name=str(id(tree.right)))
            
            # Create right child node
            self.visualize_tree(tree.right, dot=dot, parent_name=str(id(tree)), yes_name=str(id(tree.left)), no_name=str(id(tree.right)))
        
        return dot

# Classe pour Decision Trees
class DecisionTreeAlgorithm:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def apply(self, target_column: str, min_samples_split: int, max_depth: int):
        
        features = self.data.drop(columns=[target_column])
        target = self.data[target_column]

        # Conversion en tableaux numpy
        X = features.values
        Y = target.values
        Y = Y.reshape(-1,1)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

        # Mesurer le temps d'exécution pour l'entraînement
        start_time = time.time()

        regressor = DecisionTreeRegressor(min_samples_split=min_samples_split, max_depth=max_depth)
        regressor.fit(X_train,Y_train)

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Temps d'entraînement : {training_time:.4f} secondes")

        # Pour visualiser l'arbre de décision
        dot = regressor.visualize_tree()
        dot.render("decision_tree", format="png", cleanup=True)  # Sauve l'arbre au format PNG

        return regressor, training_time, "decision_tree.png", X_test, Y_test
    
    def prediction(self, X_test, Y_test, model):
        start_time = time.time()

        Y_pred = model.predict(X_test) 

        end_time = time.time()
        prediction_time = end_time - start_time
        print(f"Temps de prédiction : {prediction_time:.4f} secondes")

        MSE = np.sqrt(mean_squared_error(Y_test, Y_pred))

        return prediction_time, Y_pred, MSE
    
    def predict_value(self, X_new, model, feature_names):

        # Load min-max values from the CSV file
        min_max_df = pd.read_csv("min_max_features.csv")
        
        # Ensure the CSV file contains the required features
        if not all(feature in min_max_df.columns for feature in ['Feature', 'Min', 'Max']):
            raise ValueError("The CSV file must contain columns: 'feature', 'min', 'max'.")
        
        # Convert to a dictionary for quick lookup
        min_max_dict = min_max_df.set_index('Feature')[['Min', 'Max']].to_dict(orient='index')

        # Normalize X_new using min-max normalization
        X_normalized = []

        for value, feature in zip(X_new, feature_names):
            print(value)
            print(feature)
            min_value = min_max_dict[feature]['Min']
            max_value = min_max_dict[feature]['Max']
            
            # Min-max normalization formula
            normalized_value = (value - min_value) / (max_value - min_value)
            X_normalized.append(normalized_value)
        
        X_to_predict = X_normalized
        # Prepare input for prediction
        X_normalized = [X_normalized]  # Model expects 2D array-like input
        
        # Perform prediction
        prediction = model.predict(X_normalized)
        return prediction, X_to_predict
        

