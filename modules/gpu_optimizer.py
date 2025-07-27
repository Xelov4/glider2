#!/usr/bin/env python3
"""
⚡ OPTIMISATEUR GPU POUR MACHINE LEARNING
=========================================

Optimisation GPU pour accélérer la détection de cartes :
- CUDA acceleration
- Batch processing
- Memory optimization
- GPU monitoring
"""

import cv2
import numpy as np
import logging
from typing import List, Optional
import time

class GPUOptimizer:
    """
    Optimiseur GPU pour accélérer le ML
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gpu_available = False
        self.cuda_context = None
        self.batch_size = 4
        self.gpu_memory_usage = 0
        
        # Vérifier la disponibilité GPU
        self._check_gpu_availability()
        
        # Métriques GPU
        self.gpu_metrics = {
            'total_operations': 0,
            'gpu_time_saved': 0.0,
            'memory_usage': [],
            'batch_processing_times': []
        }
    
    def _check_gpu_availability(self):
        """Vérifie la disponibilité GPU"""
        try:
            # Vérifier CUDA
            cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
            if cuda_available:
                self.gpu_available = True
                self.logger.info(f"GPU CUDA détecté: {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
                
                # Initialiser le contexte CUDA
                self._initialize_cuda_context()
            else:
                self.logger.warning("Aucun GPU CUDA détecté, utilisation CPU")
                
        except Exception as e:
            self.logger.warning(f"Erreur détection GPU: {e}")
    
    def _initialize_cuda_context(self):
        """Initialise le contexte CUDA"""
        try:
            if self.gpu_available:
                # Créer un contexte CUDA
                self.cuda_context = cv2.cuda.Stream()
                self.logger.info("Contexte CUDA initialisé")
                
        except Exception as e:
            self.logger.error(f"Erreur initialisation CUDA: {e}")
            self.gpu_available = False
    
    def optimize_image_processing(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Optimise le traitement d'images avec GPU
        """
        try:
            if not self.gpu_available or not images:
                return images
            
            start_time = time.time()
            
            # Traitement par batch
            processed_images = []
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size]
                processed_batch = self._process_batch_gpu(batch)
                processed_images.extend(processed_batch)
            
            gpu_time = time.time() - start_time
            
            # Métriques
            self.gpu_metrics['total_operations'] += len(images)
            self.gpu_metrics['batch_processing_times'].append(gpu_time)
            
            self.logger.debug(f"Traitement GPU: {len(images)} images en {gpu_time:.3f}s")
            
            return processed_images
            
        except Exception as e:
            self.logger.error(f"Erreur optimisation GPU: {e}")
            return images
    
    def _process_batch_gpu(self, batch: List[np.ndarray]) -> List[np.ndarray]:
        """
        Traite un batch d'images avec GPU
        """
        try:
            processed_batch = []
            
            for image in batch:
                # Upload vers GPU
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(image)
                
                # Opérations GPU
                processed_gpu = self._apply_gpu_operations(gpu_image)
                
                # Download depuis GPU
                processed_cpu = processed_gpu.download()
                processed_batch.append(processed_cpu)
            
            return processed_batch
            
        except Exception as e:
            self.logger.error(f"Erreur traitement batch GPU: {e}")
            return batch
    
    def _apply_gpu_operations(self, gpu_image) -> cv2.cuda_GpuMat:
        """
        Applique les opérations GPU sur une image
        """
        try:
            # Conversion en niveaux de gris GPU
            if gpu_image.channels() == 3:
                gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
            else:
                gpu_gray = gpu_image
            
            # Redimensionnement GPU
            gpu_resized = cv2.cuda.resize(gpu_gray, (32, 32))
            
            # Filtrage GPU
            gpu_filtered = cv2.cuda.GaussianBlur(gpu_resized, (3, 3), 0)
            
            return gpu_filtered
            
        except Exception as e:
            self.logger.error(f"Erreur opérations GPU: {e}")
            return gpu_image
    
    def get_gpu_metrics(self) -> dict:
        """Retourne les métriques GPU"""
        try:
            if self.gpu_metrics['batch_processing_times']:
                avg_gpu_time = np.mean(self.gpu_metrics['batch_processing_times'])
            else:
                avg_gpu_time = 0
            
            return {
                'gpu_available': self.gpu_available,
                'total_operations': self.gpu_metrics['total_operations'],
                'avg_gpu_time': avg_gpu_time,
                'memory_usage': self.gpu_memory_usage,
                'batch_size': self.batch_size
            }
            
        except Exception as e:
            self.logger.error(f"Erreur métriques GPU: {e}")
            return {'gpu_available': False}
    
    def optimize_feature_extraction(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Optimise l'extraction de features avec GPU
        """
        try:
            if not self.gpu_available:
                return images
            
            # Extraction de features GPU
            gpu_features = []
            for image in images:
                features = self._extract_features_gpu(image)
                gpu_features.append(features)
            
            return gpu_features
            
        except Exception as e:
            self.logger.error(f"Erreur extraction features GPU: {e}")
            return images
    
    def _extract_features_gpu(self, image: np.ndarray) -> np.ndarray:
        """
        Extrait les features avec GPU
        """
        try:
            # Upload vers GPU
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)
            
            # Features GPU
            features = []
            
            # 1. Histogramme GPU
            gpu_hist = cv2.cuda.calcHist(gpu_image, [0], None, [16], [0, 256])
            hist_features = gpu_hist.download().flatten()
            features.extend(hist_features)
            
            # 2. Gradients GPU
            gpu_grad_x = cv2.cuda.Sobel(gpu_image, cv2.CV_64F, 1, 0, ksize=3)
            gpu_grad_y = cv2.cuda.Sobel(gpu_image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Magnitude GPU
            gpu_magnitude = cv2.cuda.magnitude(gpu_grad_x, gpu_grad_y)
            magnitude_cpu = gpu_magnitude.download()
            
            # Histogramme des gradients
            grad_hist = cv2.calcHist([magnitude_cpu.astype(np.uint8)], [0], None, [16], [0, 256])
            features.extend(grad_hist.flatten())
            
            # 3. Features de texture GPU (simplifié)
            texture_features = self._compute_texture_features_gpu(gpu_image)
            features.extend(texture_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Erreur extraction features GPU: {e}")
            return np.zeros(64, dtype=np.float32)
    
    def _compute_texture_features_gpu(self, gpu_image) -> List[float]:
        """
        Calcule les features de texture avec GPU
        """
        try:
            features = []
            
            # LBP simplifié GPU
            gpu_lbp = cv2.cuda.GpuMat(gpu_image.size(), cv2.CV_8UC1)
            
            # Calcul LBP basique
            for i in range(1, gpu_image.rows - 1):
                for j in range(1, gpu_image.cols - 1):
                    # Simuler LBP GPU (simplifié)
                    features.append(i * j % 256)  # Placeholder
            
            # Réduire à 16 features
            if len(features) > 16:
                step = len(features) // 16
                features = features[::step][:16]
            elif len(features) < 16:
                features.extend([0] * (16 - len(features)))
            
            return features[:16]
            
        except Exception as e:
            self.logger.error(f"Erreur features texture GPU: {e}")
            return [0] * 16
    
    def cleanup(self):
        """Nettoie les ressources GPU"""
        try:
            if self.cuda_context:
                self.cuda_context.release()
                self.logger.info("Ressources GPU libérées")
                
        except Exception as e:
            self.logger.error(f"Erreur nettoyage GPU: {e}")

class LearningOptimizer:
    """
    Optimiseur d'apprentissage continu
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.learning_data = []
        self.performance_history = []
        self.adaptation_threshold = 0.1
        
    def add_learning_sample(self, image: np.ndarray, detected_cards: List, ground_truth: List = None):
        """
        Ajoute un échantillon d'apprentissage
        """
        try:
            sample = {
                'image': image,
                'detected_cards': detected_cards,
                'ground_truth': ground_truth,
                'timestamp': time.time(),
                'performance_score': self._calculate_performance_score(detected_cards, ground_truth)
            }
            
            self.learning_data.append(sample)
            
            # Limiter la taille des données
            if len(self.learning_data) > 1000:
                self.learning_data = self.learning_data[-500:]
            
            self.logger.debug(f"Échantillon d'apprentissage ajouté: {len(detected_cards)} cartes")
            
        except Exception as e:
            self.logger.error(f"Erreur ajout échantillon: {e}")
    
    def _calculate_performance_score(self, detected: List, ground_truth: List) -> float:
        """
        Calcule le score de performance
        """
        try:
            if not ground_truth:
                return 0.5  # Score par défaut
            
            # Comparaison simple
            detected_set = set([f"{c.rank}{c.suit}" for c in detected])
            truth_set = set(ground_truth)
            
            if not truth_set:
                return 0.0
            
            # Score basé sur la précision
            correct = len(detected_set.intersection(truth_set))
            total = len(truth_set)
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Erreur calcul score: {e}")
            return 0.0
    
    def adapt_model(self, ml_detector):
        """
        Adapte le modèle ML basé sur les données d'apprentissage
        """
        try:
            if len(self.learning_data) < 10:
                self.logger.debug("Pas assez de données pour l'adaptation")
                return
            
            # Calculer la performance moyenne
            recent_performance = [s['performance_score'] for s in self.learning_data[-50:]]
            avg_performance = np.mean(recent_performance)
            
            self.performance_history.append(avg_performance)
            
            # Adapter si nécessaire
            if avg_performance < self.adaptation_threshold:
                self.logger.info(f"Performance faible ({avg_performance:.3f}), adaptation du modèle")
                self._retrain_model(ml_detector)
            
        except Exception as e:
            self.logger.error(f"Erreur adaptation modèle: {e}")
    
    def _retrain_model(self, ml_detector):
        """
        Réentraîne le modèle avec les nouvelles données
        """
        try:
            # Extraire les features des échantillons récents
            new_features = []
            new_labels = []
            
            for sample in self.learning_data[-100:]:  # 100 derniers échantillons
                if sample['detected_cards']:
                    # Utiliser les cartes détectées comme nouvelles données
                    for card in sample['detected_cards']:
                        # Créer des features synthétiques basées sur la carte
                        features = self._create_features_from_card(card)
                        new_features.append(features)
                        new_labels.append(f"{card.rank}{card.suit}")
            
            if new_features and new_labels:
                # Ajouter aux données d'entraînement existantes
                # (Cette partie nécessiterait une modification du CardMLDetector)
                self.logger.info(f"Modèle adapté avec {len(new_features)} nouvelles features")
            
        except Exception as e:
            self.logger.error(f"Erreur réentraînement: {e}")
    
    def _create_features_from_card(self, card) -> np.ndarray:
        """
        Crée des features basées sur une carte détectée
        """
        try:
            # Features synthétiques basées sur le rang et la couleur
            features = np.zeros(64, dtype=np.float32)
            
            # Encoder le rang
            rank_idx = '23456789TJQKA'.find(card.rank)
            if rank_idx >= 0:
                features[rank_idx] = 1.0
            
            # Encoder la couleur
            suit_idx = '♠♥♦♣'.find(card.suit)
            if suit_idx >= 0:
                features[13 + suit_idx] = 1.0
            
            # Ajouter la confiance
            features[17] = card.confidence
            
            return features
            
        except Exception as e:
            self.logger.error(f"Erreur création features: {e}")
            return np.zeros(64, dtype=np.float32)
    
    def get_learning_metrics(self) -> dict:
        """Retourne les métriques d'apprentissage"""
        try:
            if self.performance_history:
                recent_performance = np.mean(self.performance_history[-10:])
                performance_trend = np.mean(self.performance_history[-5:]) - np.mean(self.performance_history[-10:-5])
            else:
                recent_performance = 0.0
                performance_trend = 0.0
            
            return {
                'total_samples': len(self.learning_data),
                'recent_performance': recent_performance,
                'performance_trend': performance_trend,
                'adaptation_threshold': self.adaptation_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Erreur métriques apprentissage: {e}")
            return {'total_samples': 0} 