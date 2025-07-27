#!/usr/bin/env python3
"""
🧠 DÉTECTEUR DE CARTES PAR MACHINE LEARNING
============================================

Système de reconnaissance de cartes basé sur ML simple :
- Template matching avec features
- Classification par similarité
- Apprentissage automatique des patterns
- Détection robuste des couleurs
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import pickle
import os

@dataclass
class CardFeature:
    """Features d'une carte pour ML"""
    rank: str
    suit: str
    features: np.ndarray
    confidence: float

class CardMLDetector:
    """
    Détecteur de cartes par Machine Learning
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.knn_classifier = None
        self.feature_database = {}
        self.color_clusters = None
        self.is_trained = False
        
        # Configuration ML
        self.feature_size = (32, 32)  # Taille des features
        self.n_neighbors = 3  # KNN
        self.color_threshold = 0.1  # Seuil de couleur
        
        # Initialiser le modèle
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialise le modèle ML"""
        try:
            # Créer le classifieur KNN
            self.knn_classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            
            # Créer la base de données de features
            self._create_feature_database()
            
            # Entraîner le modèle
            self._train_model()
            
            self.logger.info("Modèle ML initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur initialisation ML: {e}")
    
    def _create_feature_database(self):
        """Crée la base de données de features pour chaque carte"""
        try:
            # NOUVEAU: Features synthétiques pour chaque carte
            ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
            suits = ['♠', '♥', '♦', '♣']
            
            for rank in ranks:
                for suit in suits:
                    card_name = f"{rank}{suit}"
                    
                    # Créer des features synthétiques basées sur le rang et la couleur
                    features = self._generate_card_features(rank, suit)
                    
                    self.feature_database[card_name] = {
                        'rank': rank,
                        'suit': suit,
                        'features': features,
                        'color_type': 'red' if suit in ['♥', '♦'] else 'black'
                    }
            
            self.logger.debug(f"Base de données créée: {len(self.feature_database)} cartes")
            
        except Exception as e:
            self.logger.error(f"Erreur création base de données: {e}")
    
    def _generate_card_features(self, rank: str, suit: str) -> np.ndarray:
        """Génère des features synthétiques pour une carte"""
        try:
            # Créer une image synthétique de la carte
            card_img = np.ones(self.feature_size, dtype=np.uint8) * 255  # Fond blanc
            
            # Ajouter le rang (pattern basé sur le caractère)
            rank_pattern = self._get_rank_pattern(rank)
            card_img[5:27, 5:27] = rank_pattern
            
            # Ajouter la couleur
            if suit in ['♥', '♦']:
                # Rouge
                card_img[20:30, 5:27] = 100  # Zone rouge
            else:
                # Noir
                card_img[20:30, 5:27] = 50   # Zone noire
            
            # Extraire les features
            features = self._extract_features(card_img)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Erreur génération features: {e}")
            return np.zeros(64)  # Features par défaut
    
    def _get_rank_pattern(self, rank: str) -> np.ndarray:
        """Génère un pattern pour un rang"""
        pattern = np.zeros((22, 22), dtype=np.uint8)
        
        # Patterns basés sur le rang
        if rank == 'A':
            # As - pattern en A
            pattern[5:17, 8:14] = 255
            pattern[8:14, 5:17] = 255
        elif rank == 'K':
            # Roi - pattern en K
            pattern[5:17, 5:17] = 255
            pattern[8:14, 8:14] = 0
        elif rank == 'Q':
            # Reine - pattern en Q
            pattern[5:17, 5:17] = 255
            pattern[8:14, 8:14] = 0
        elif rank == 'J':
            # Valet - pattern en J
            pattern[5:17, 5:17] = 255
            pattern[8:14, 8:14] = 0
        elif rank == 'T':
            # 10 - pattern en T
            pattern[5:17, 5:17] = 255
            pattern[8:14, 8:14] = 0
        else:
            # Chiffres - pattern simple
            pattern[5:17, 5:17] = 255
        
        return pattern
    
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extrait les features d'une image"""
        try:
            # Redimensionner à une taille fixe
            resized = cv2.resize(image, self.feature_size)
            
            # Conversion en niveaux de gris si nécessaire
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            
            # Features basiques : histogramme + gradients
            features = []
            
            # 1. Histogramme de niveaux de gris
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            features.extend(hist.flatten())
            
            # 2. Gradients (Sobel)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Histogramme des gradients
            grad_hist = cv2.calcHist([gradient_magnitude.astype(np.uint8)], [0], None, [16], [0, 256])
            features.extend(grad_hist.flatten())
            
            # 3. Features de texture (LBP simplifié)
            lbp_features = self._compute_lbp_features(gray)
            features.extend(lbp_features)
            
            # 4. Features de forme
            shape_features = self._compute_shape_features(gray)
            features.extend(shape_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Erreur extraction features: {e}")
            return np.zeros(64, dtype=np.float32)
    
    def _compute_lbp_features(self, image: np.ndarray) -> List[float]:
        """Calcule les features LBP (Local Binary Pattern) simplifiées"""
        try:
            features = []
            
            # LBP simplifié
            for i in range(1, image.shape[0]-1):
                for j in range(1, image.shape[1]-1):
                    center = image[i, j]
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j-1], image[i, j+1],
                        image[i+1, j-1], image[i+1, j], image[i+1, j+1]
                    ]
                    
                    # Calculer le pattern binaire
                    pattern = sum(1 << idx for idx, neighbor in enumerate(neighbors) if neighbor >= center)
                    features.append(pattern)
            
            # Réduire à 16 features
            if len(features) > 16:
                step = len(features) // 16
                features = features[::step][:16]
            elif len(features) < 16:
                features.extend([0] * (16 - len(features)))
            
            return features[:16]
            
        except Exception as e:
            self.logger.error(f"Erreur LBP: {e}")
            return [0] * 16
    
    def _compute_shape_features(self, image: np.ndarray) -> List[float]:
        """Calcule les features de forme"""
        try:
            features = []
            
            # 1. Densité de pixels
            total_pixels = image.shape[0] * image.shape[1]
            white_pixels = np.sum(image > 127)
            density = white_pixels / total_pixels
            features.append(density)
            
            # 2. Contours
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            features.append(len(contours))
            
            # 3. Aire du plus grand contour
            if contours:
                max_area = max(cv2.contourArea(c) for c in contours)
                features.append(max_area / total_pixels)
            else:
                features.append(0)
            
            # 4. Ratio d'aspect
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                aspect_ratio = w / h if h > 0 else 0
                features.append(aspect_ratio)
            else:
                features.append(0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Erreur features forme: {e}")
            return [0] * 4
    
    def _train_model(self):
        """Entraîne le modèle ML"""
        try:
            # Préparer les données d'entraînement
            X_train = []
            y_train = []
            
            for card_name, card_data in self.feature_database.items():
                X_train.append(card_data['features'])
                y_train.append(card_name)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Entraîner le classifieur KNN
            self.knn_classifier.fit(X_train, y_train)
            
            # Entraîner les clusters de couleur
            self._train_color_clusters()
            
            self.is_trained = True
            self.logger.info(f"Modèle entraîné: {len(X_train)} échantillons")
            
        except Exception as e:
            self.logger.error(f"Erreur entraînement: {e}")
    
    def _train_color_clusters(self):
        """Entraîne les clusters de couleur"""
        try:
            # Créer des échantillons de couleur
            color_samples = []
            
            for card_name, card_data in self.feature_database.items():
                if card_data['color_type'] == 'red':
                    color_samples.append([255, 0, 0])  # Rouge
                else:
                    color_samples.append([0, 0, 0])    # Noir
            
            if color_samples:
                color_samples = np.array(color_samples)
                self.color_clusters = KMeans(n_clusters=2, random_state=42)
                self.color_clusters.fit(color_samples)
            
        except Exception as e:
            self.logger.error(f"Erreur clusters couleur: {e}")
    
    def detect_cards_ml(self, image: np.ndarray) -> List[CardFeature]:
        """
        Détecte les cartes avec ML
        """
        try:
            if not self.is_trained:
                self.logger.warning("Modèle non entraîné")
                return []
            
            cards = []
            
            # 1. Détecter les régions de cartes
            card_regions = self._detect_card_regions(image)
            
            # 2. Analyser chaque région
            for region in card_regions:
                card = self._analyze_card_region_ml(region)
                if card:
                    cards.append(card)
            
            return cards
            
        except Exception as e:
            self.logger.error(f"Erreur détection ML: {e}")
            return []
    
    def _detect_card_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """Détecte les régions de cartes dans l'image"""
        try:
            regions = []
            
            # Conversion en niveaux de gris
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Seuillage adaptatif
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Recherche de contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:  # Filtre les petits contours
                    # Vérifier le ratio d'aspect
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if 0.5 < aspect_ratio < 2.0:  # Ratio acceptable pour une carte
                        # Extraire la région
                        margin = 5
                        x = max(0, x - margin)
                        y = max(0, y - margin)
                        w = min(w + 2 * margin, image.shape[1] - x)
                        h = min(h + 2 * margin, image.shape[0] - y)
                        
                        region = image[y:y+h, x:x+w]
                        if region.size > 0:
                            regions.append(region)
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Erreur détection régions: {e}")
            return []
    
    def _analyze_card_region_ml(self, region: np.ndarray) -> Optional[CardFeature]:
        """Analyse une région de carte avec ML"""
        try:
            # Extraire les features
            features = self._extract_features(region)
            
            # Prédire la carte avec KNN
            prediction = self.knn_classifier.predict([features])[0]
            confidence = self.knn_classifier.predict_proba([features]).max()
            
            # Analyser la couleur
            color_analysis = self._analyze_color_ml(region)
            
            # Créer l'objet carte
            rank, suit = prediction[:-1], prediction[-1]
            
            card = CardFeature(
                rank=rank,
                suit=suit,
                features=features,
                confidence=confidence
            )
            
            self.logger.debug(f"Carte ML détectée: {rank}{suit} (conf: {confidence:.2f})")
            return card
            
        except Exception as e:
            self.logger.error(f"Erreur analyse ML: {e}")
            return None
    
    def _analyze_color_ml(self, region: np.ndarray) -> Dict:
        """Analyse la couleur avec ML"""
        try:
            # Conversion HSV
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Masques de couleur
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 50])
            mask_black = cv2.inRange(hsv, lower_black, upper_black)
            
            # Compter les pixels
            red_pixels = cv2.countNonZero(mask_red)
            black_pixels = cv2.countNonZero(mask_black)
            total_pixels = region.shape[0] * region.shape[1]
            
            return {
                'red_ratio': red_pixels / total_pixels if total_pixels > 0 else 0,
                'black_ratio': black_pixels / total_pixels if total_pixels > 0 else 0,
                'dominant_color': 'red' if red_pixels > black_pixels else 'black'
            }
            
        except Exception as e:
            self.logger.error(f"Erreur analyse couleur ML: {e}")
            return {'red_ratio': 0, 'black_ratio': 0, 'dominant_color': 'unknown'}
    
    def save_model(self, filepath: str = "card_ml_model.pkl"):
        """Sauvegarde le modèle ML"""
        try:
            model_data = {
                'knn_classifier': self.knn_classifier,
                'feature_database': self.feature_database,
                'color_clusters': self.color_clusters,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Modèle sauvegardé: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde modèle: {e}")
    
    def load_model(self, filepath: str = "card_ml_model.pkl"):
        """Charge le modèle ML"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.knn_classifier = model_data['knn_classifier']
                self.feature_database = model_data['feature_database']
                self.color_clusters = model_data['color_clusters']
                self.is_trained = model_data['is_trained']
                
                self.logger.info(f"Modèle chargé: {filepath}")
            else:
                self.logger.warning(f"Fichier modèle non trouvé: {filepath}")
                
        except Exception as e:
            self.logger.error(f"Erreur chargement modèle: {e}") 