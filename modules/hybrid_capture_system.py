#!/usr/bin/env python3
"""
🔧 SYSTÈME HYBRIDE DE CAPTURE
===============================

Système hybride : Capture maximale + Post-traitement intelligent
pour optimiser la détection de cartes.
"""

import cv2
import numpy as np
import time
import json
import os
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import pyautogui
from PIL import Image, ImageEnhance, ImageFilter

@dataclass
class CaptureConfig:
    """Configuration de capture hybride"""
    capture_method: str = "max_quality"  # max_quality, multi_scale, adaptive
    post_processing: bool = True
    upscale_factor: float = 3.0  # Facteur d'agrandissement
    enhancement_strength: float = 1.5
    noise_reduction: bool = True
    sharpening: bool = True
    contrast_boost: bool = True

class HybridCaptureSystem:
    """
    Système hybride de capture maximale + post-traitement
    """
    
    def __init__(self, config: CaptureConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or CaptureConfig()
        
        # Charger les régions
        self.regions = self._load_regions()
        
        # Cache pour optimiser
        self.capture_cache = {}
        self.cache_duration = 0.1
        
        # Métriques de performance
        self.performance_metrics = {
            'total_captures': 0,
            'avg_capture_time': 0.0,
            'avg_quality_score': 0.0,
            'post_processing_times': [],
            'quality_improvements': []
        }
    
    def _load_regions(self) -> Dict:
        """Charge les régions calibrées"""
        try:
            if os.path.exists("calibrated_regions.json"):
                with open("calibrated_regions.json", 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning("Fichier calibrated_regions.json non trouvé")
                return {}
        except Exception as e:
            self.logger.error(f"Erreur chargement régions: {e}")
            return {}
    
    def capture_with_max_quality(self, region_name: str) -> Optional[np.ndarray]:
        """
        Capture avec qualité maximale + post-traitement
        """
        try:
            if region_name not in self.regions:
                self.logger.warning(f"Région inconnue: {region_name}")
                return None
            
            region_data = self.regions[region_name]
            start_time = time.time()
            
            # 1. CAPTURE MAXIMALE
            raw_image = self._capture_max_quality(region_data)
            if raw_image is None:
                return None
            
            capture_time = time.time() - start_time
            
            # 2. POST-TRAITEMENT INTELLIGENT
            if self.config.post_processing:
                processed_image = self._post_process_for_cards(raw_image, region_name)
                post_time = time.time() - start_time - capture_time
                
                self.performance_metrics['post_processing_times'].append(post_time)
                
                # Calculer l'amélioration
                raw_quality = self._calculate_quality_score(raw_image)
                processed_quality = self._calculate_quality_score(processed_image)
                improvement = ((processed_quality - raw_quality) / raw_quality) * 100
                self.performance_metrics['quality_improvements'].append(improvement)
                
                self.logger.debug(f"Capture {region_name}: {raw_image.shape} → {processed_image.shape}, amélioration: +{improvement:.1f}%")
                
                return processed_image
            else:
                return raw_image
            
        except Exception as e:
            self.logger.error(f"Erreur capture hybride {region_name}: {e}")
            return None
    
    def _capture_max_quality(self, region_data: Dict) -> Optional[np.ndarray]:
        """
        Capture avec qualité maximale possible
        """
        try:
            # Méthode 1: Capture standard optimisée
            screenshot = pyautogui.screenshot(
                region=(region_data['x'], region_data['y'], region_data['width'], region_data['height'])
            )
            
            # Conversion en numpy avec haute qualité
            image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # NOUVEAU: Capture multi-résolution si l'image est petite
            if image.shape[1] < 400:  # Largeur trop petite
                # Essayer de capturer une région plus grande
                expanded_region = self._expand_region(region_data)
                if expanded_region:
                    expanded_screenshot = pyautogui.screenshot(region=expanded_region)
                    expanded_image = cv2.cvtColor(np.array(expanded_screenshot), cv2.COLOR_RGB2BGR)
                    
                    # Redimensionner à la taille originale
                    target_size = (region_data['width'], region_data['height'])
                    expanded_image = cv2.resize(expanded_image, target_size, interpolation=cv2.INTER_CUBIC)
                    
                    # Utiliser l'image étendue si elle est meilleure
                    if self._calculate_quality_score(expanded_image) > self._calculate_quality_score(image):
                        image = expanded_image
                        self.logger.debug(f"Image étendue utilisée pour meilleure qualité")
            
            return image
            
        except Exception as e:
            self.logger.error(f"Erreur capture max qualité: {e}")
            return None
    
    def _expand_region(self, region_data: Dict) -> Optional[Tuple]:
        """
        Étend la région de capture pour obtenir plus de détails
        """
        try:
            x, y, w, h = region_data['x'], region_data['y'], region_data['width'], region_data['height']
            
            # Étendre de 50% dans chaque direction
            expansion = 0.5
            new_x = max(0, int(x - w * expansion))
            new_y = max(0, int(y - h * expansion))
            new_w = int(w * (1 + 2 * expansion))
            new_h = int(h * (1 + 2 * expansion))
            
            return (new_x, new_y, new_w, new_h)
            
        except Exception as e:
            self.logger.error(f"Erreur expansion région: {e}")
            return None
    
    def _post_process_for_cards(self, image: np.ndarray, region_name: str) -> np.ndarray:
        """
        Post-traitement intelligent pour la détection de cartes
        """
        try:
            processed = image.copy()
            
            # 1. UPSCALING AGGRESSIF POUR LES CARTES
            if region_name in ['hand_area', 'community_cards']:
                height, width = processed.shape[:2]
                if width < 600:  # Si l'image est trop petite
                    scale_factor = self.config.upscale_factor
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    
                    # Upscaling avec interpolation de haute qualité
                    processed = cv2.resize(processed, (new_width, new_height), 
                                        interpolation=cv2.INTER_CUBIC)
                    
                    self.logger.debug(f"Upscaling {region_name}: {width}x{height} → {new_width}x{new_height}")
            
            # 2. AMÉLIORATION DU CONTRASTE AVANCÉE
            if self.config.contrast_boost:
                processed = self._enhance_contrast_advanced(processed)
            
            # 3. RÉDUCTION DU BRUIT INTELLIGENTE
            if self.config.noise_reduction:
                processed = self._reduce_noise_intelligent(processed)
            
            # 4. SHARPENING SPÉCIFIQUE AUX CARTES
            if self.config.sharpening:
                processed = self._sharpen_for_cards(processed)
            
            # 5. OPTIMISATION DES COULEURS
            processed = self._optimize_colors_for_cards(processed)
            
            # 6. AMÉLIORATION DES SYMBOLES
            processed = self._enhance_card_symbols(processed)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Erreur post-traitement: {e}")
            return image
    
    def _enhance_contrast_advanced(self, image: np.ndarray) -> np.ndarray:
        """
        Amélioration avancée du contraste
        """
        try:
            # Conversion LAB pour meilleur contrôle
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # CLAHE sur le canal L avec paramètres optimisés
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            
            # Amélioration des canaux a et b (couleurs)
            lab[:,:,1] = cv2.multiply(lab[:,:,1], 1.2)  # Rouge-vert
            lab[:,:,2] = cv2.multiply(lab[:,:,2], 1.2)  # Bleu-jaune
            
            # Reconversion BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Erreur amélioration contraste: {e}")
            return image
    
    def _reduce_noise_intelligent(self, image: np.ndarray) -> np.ndarray:
        """
        Réduction intelligente du bruit
        """
        try:
            # Filtre bilatéral pour préserver les bords
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Filtre médian pour réduire le bruit de sel et poivre
            denoised = cv2.medianBlur(denoised, 3)
            
            # Filtre gaussien léger pour lisser
            denoised = cv2.GaussianBlur(denoised, (3, 3), 0.5)
            
            return denoised
            
        except Exception as e:
            self.logger.error(f"Erreur réduction bruit: {e}")
            return image
    
    def _sharpen_for_cards(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpening spécifique pour les cartes
        """
        try:
            # Kernel de sharpening optimisé pour les symboles de cartes
            strength = self.config.enhancement_strength
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]]) * strength
            
            # Application du filtre
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Combiner avec l'original pour éviter l'excès
            enhanced = cv2.addWeighted(image, 0.6, sharpened, 0.4, 0)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Erreur sharpening: {e}")
            return image
    
    def _optimize_colors_for_cards(self, image: np.ndarray) -> np.ndarray:
        """
        Optimisation des couleurs pour les cartes
        """
        try:
            # Conversion HSV pour meilleur contrôle
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Augmenter la saturation (couleurs des cartes)
            hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.4)
            
            # Ajuster la valeur (luminosité)
            hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.3)
            
            # Reconversion BGR
            optimized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return optimized
            
        except Exception as e:
            self.logger.error(f"Erreur optimisation couleurs: {e}")
            return image
    
    def _enhance_card_symbols(self, image: np.ndarray) -> np.ndarray:
        """
        Amélioration spécifique des symboles de cartes
        """
        try:
            # Conversion en niveaux de gris
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Seuillage adaptatif pour isoler les symboles
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphologie pour nettoyer et renforcer
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            enhanced = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Dilatation pour renforcer les symboles
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            enhanced = cv2.dilate(enhanced, kernel_dilate, iterations=1)
            
            # Combiner avec l'image originale
            if len(image.shape) == 3:
                enhanced_3d = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                result = cv2.addWeighted(image, 0.7, enhanced_3d, 0.3, 0)
            else:
                result = cv2.addWeighted(image, 0.7, enhanced, 0.3, 0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur amélioration symboles: {e}")
            return image
    
    def _calculate_quality_score(self, image: np.ndarray) -> float:
        """
        Calcule un score de qualité d'image
        """
        try:
            score = 0.0
            
            # 1. Résolution (plus important)
            height, width = image.shape[:2]
            resolution_score = min(width * height / 10000, 1.0)
            score += resolution_score * 0.4
            
            # 2. Contraste
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            contrast_score = np.std(gray) / 255.0
            score += contrast_score * 0.3
            
            # 3. Netteté
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            sharpness_score = np.mean(gradient_magnitude) / 255.0
            score += sharpness_score * 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Erreur calcul qualité: {e}")
            return 0.5
    
    def get_performance_metrics(self) -> dict:
        """Retourne les métriques de performance"""
        try:
            if self.performance_metrics['post_processing_times']:
                avg_post_time = np.mean(self.performance_metrics['post_processing_times'])
                avg_improvement = np.mean(self.performance_metrics['quality_improvements'])
            else:
                avg_post_time = 0.0
                avg_improvement = 0.0
            
            return {
                'total_captures': self.performance_metrics['total_captures'],
                'avg_capture_time': self.performance_metrics['avg_capture_time'],
                'avg_post_processing_time': avg_post_time,
                'avg_quality_improvement': avg_improvement
            }
            
        except Exception as e:
            self.logger.error(f"Erreur métriques performance: {e}")
            return {'total_captures': 0}
    
    def set_config(self, config: CaptureConfig):
        """Met à jour la configuration"""
        self.config = config
        self.logger.info(f"Configuration mise à jour: {config}") 