#!/usr/bin/env python3
"""
📸 MODULE DE CAPTURE HAUTE QUALITÉ
===================================

Système de capture d'écran optimisé pour la détection de cartes :
- Résolution haute qualité
- Upscaling intelligent
- Prétraitement optimisé
- Multi-méthodes de capture
"""

import cv2
import numpy as np
import pyautogui
import time
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter
import mss
import mss.tools

@dataclass
class CaptureConfig:
    """Configuration de capture haute qualité"""
    resolution_scale: float = 2.0  # Facteur d'échelle pour la résolution
    quality_enhancement: bool = True  # Amélioration de la qualité
    sharpening: bool = True  # Netteté
    contrast_enhancement: bool = True  # Amélioration du contraste
    noise_reduction: bool = True  # Réduction du bruit
    capture_method: str = "mss"  # mss, pyautogui, or pil

class HighQualityCapture:
    """
    Module de capture haute qualité pour détection de cartes
    """
    
    def __init__(self, config: CaptureConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or CaptureConfig()
        
        # Charger les régions calibrées
        self.regions = self._load_calibrated_regions()
        
        # Initialiser MSS pour capture haute qualité
        self.mss_instance = None
        try:
            self.mss_instance = mss.mss()
            self.logger.info("MSS initialisé pour capture haute qualité")
        except Exception as e:
            self.logger.warning(f"MSS non disponible: {e}")
        
        # Métriques de qualité
        self.quality_metrics = {
            'total_captures': 0,
            'avg_resolution': 0,
            'quality_scores': [],
            'capture_times': []
        }
    
    def _load_calibrated_regions(self) -> Dict:
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
    
    def capture_region_high_quality(self, region_name: str) -> Optional[np.ndarray]:
        """
        Capture une région avec haute qualité
        """
        try:
            if region_name not in self.regions:
                self.logger.warning(f"Région inconnue: {region_name}")
                return None
            
            region_data = self.regions[region_name]
            start_time = time.time()
            
            # Capture selon la méthode configurée
            if self.config.capture_method == "mss" and self.mss_instance:
                image = self._capture_with_mss(region_data)
            elif self.config.capture_method == "pil":
                image = self._capture_with_pil(region_data)
            else:
                image = self._capture_with_pyautogui(region_data)
            
            if image is None:
                return None
            
            # Amélioration de la qualité
            enhanced_image = self._enhance_image_quality(image)
            
            # Métriques
            capture_time = time.time() - start_time
            self.quality_metrics['total_captures'] += 1
            self.quality_metrics['capture_times'].append(capture_time)
            
            # Calculer le score de qualité
            quality_score = self._calculate_quality_score(enhanced_image)
            self.quality_metrics['quality_scores'].append(quality_score)
            
            self.logger.debug(f"Capture {region_name}: {enhanced_image.shape}, qualité: {quality_score:.2f}")
            
            return enhanced_image
            
        except Exception as e:
            self.logger.error(f"Erreur capture haute qualité {region_name}: {e}")
            return None
    
    def _capture_with_mss(self, region_data: Dict) -> Optional[np.ndarray]:
        """
        Capture avec MSS (plus rapide et haute qualité)
        """
        try:
            # MSS capture
            with self.mss_instance.grab(region_data) as screenshot:
                # Convertir en PIL Image
                pil_image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                
                # Convertir en numpy array
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                return image
                
        except Exception as e:
            self.logger.error(f"Erreur capture MSS: {e}")
            return None
    
    def _capture_with_pil(self, region_data: Dict) -> Optional[np.ndarray]:
        """
        Capture avec PIL (haute qualité)
        """
        try:
            # Capture avec PIL
            pil_image = Image.grab(bbox=(
                region_data['x'], 
                region_data['y'], 
                region_data['x'] + region_data['width'],
                region_data['y'] + region_data['height']
            ))
            
            # Convertir en numpy array
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Erreur capture PIL: {e}")
            return None
    
    def _capture_with_pyautogui(self, region_data: Dict) -> Optional[np.ndarray]:
        """
        Capture avec pyautogui (méthode de fallback)
        """
        try:
            # Capture pyautogui
            screenshot = pyautogui.screenshot(region=(
                region_data['x'],
                region_data['y'],
                region_data['width'],
                region_data['height']
            ))
            
            # Convertir en numpy array
            image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Erreur capture pyautogui: {e}")
            return None
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Améliore la qualité de l'image pour la détection de cartes
        """
        try:
            enhanced = image.copy()
            
            # 1. UPSCALING INTELLIGENT
            if self.config.resolution_scale > 1.0:
                height, width = enhanced.shape[:2]
                new_width = int(width * self.config.resolution_scale)
                new_height = int(height * self.config.resolution_scale)
                
                # Upscaling avec interpolation de haute qualité
                enhanced = cv2.resize(enhanced, (new_width, new_height), 
                                   interpolation=cv2.INTER_CUBIC)
            
            # 2. AMÉLIORATION DU CONTRASTE
            if self.config.contrast_enhancement:
                # CLAHE pour améliorer le contraste local
                if len(enhanced.shape) == 3:
                    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    lab[:,:,0] = clahe.apply(lab[:,:,0])
                    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                else:
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(enhanced)
            
            # 3. NETTETÉ
            if self.config.sharpening:
                # Kernel de netteté
                kernel = np.array([[-1,-1,-1],
                                 [-1, 9,-1],
                                 [-1,-1,-1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # 4. RÉDUCTION DU BRUIT
            if self.config.noise_reduction:
                # Filtre bilatéral pour réduire le bruit tout en préservant les bords
                enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # 5. AMÉLIORATION DE LA LUMINOSITÉ
            if self.config.quality_enhancement:
                # Ajustement gamma pour améliorer la visibilité
                gamma = 1.2
                enhanced = np.power(enhanced / 255.0, gamma) * 255.0
                enhanced = enhanced.astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Erreur amélioration qualité: {e}")
            return image
    
    def _calculate_quality_score(self, image: np.ndarray) -> float:
        """
        Calcule un score de qualité de l'image
        """
        try:
            score = 0.0
            
            # 1. Résolution
            height, width = image.shape[:2]
            resolution_score = min(width * height / 10000, 1.0)  # Normalisé
            score += resolution_score * 0.3
            
            # 2. Contraste
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            contrast_score = np.std(gray) / 255.0  # Écart-type normalisé
            score += contrast_score * 0.3
            
            # 3. Netteté (gradient)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            sharpness_score = np.mean(gradient_magnitude) / 255.0
            score += sharpness_score * 0.2
            
            # 4. Bruit (inverse)
            noise_score = 1.0 - (np.std(gray) / np.mean(gray)) if np.mean(gray) > 0 else 0.0
            score += noise_score * 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Erreur calcul score qualité: {e}")
            return 0.5
    
    def get_quality_metrics(self) -> Dict:
        """Retourne les métriques de qualité"""
        try:
            if self.quality_metrics['quality_scores']:
                avg_quality = np.mean(self.quality_metrics['quality_scores'])
                avg_capture_time = np.mean(self.quality_metrics['capture_times'])
            else:
                avg_quality = 0.0
                avg_capture_time = 0.0
            
            return {
                'total_captures': self.quality_metrics['total_captures'],
                'avg_quality_score': avg_quality,
                'avg_capture_time': avg_capture_time,
                'resolution_scale': self.config.resolution_scale,
                'capture_method': self.config.capture_method
            }
            
        except Exception as e:
            self.logger.error(f"Erreur métriques qualité: {e}")
            return {'total_captures': 0}
    
    def optimize_for_card_detection(self, region_name: str) -> Optional[np.ndarray]:
        """
        Optimise spécifiquement pour la détection de cartes
        """
        try:
            # Capture haute qualité
            image = self.capture_region_high_quality(region_name)
            if image is None:
                return None
            
            # Optimisations spécifiques aux cartes
            optimized = self._optimize_for_cards(image)
            
            return optimized
            
        except Exception as e:
            self.logger.error(f"Erreur optimisation cartes: {e}")
            return None
    
    def _optimize_for_cards(self, image: np.ndarray) -> np.ndarray:
        """
        Optimise l'image spécifiquement pour la détection de cartes
        """
        try:
            optimized = image.copy()
            
            # 1. REDIMENSIONNEMENT OPTIMAL POUR LES CARTES
            # Les cartes ont généralement un ratio d'aspect de 2.5:3.5
            height, width = optimized.shape[:2]
            target_width = max(width, 200)  # Largeur minimale
            target_height = int(target_width * 1.4)  # Ratio carte
            
            optimized = cv2.resize(optimized, (target_width, target_height), 
                                 interpolation=cv2.INTER_CUBIC)
            
            # 2. AMÉLIORATION DU CONTRASTE SPÉCIFIQUE AUX CARTES
            if len(optimized.shape) == 3:
                # Conversion HSV pour meilleur contrôle
                hsv = cv2.cvtColor(optimized, cv2.COLOR_BGR2HSV)
                
                # Augmenter la saturation (couleurs des cartes)
                hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.3)
                
                # Ajuster la valeur (luminosité)
                hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.2)
                
                optimized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # 3. FILTRE SPÉCIFIQUE POUR LES SYMBOLES DE CARTES
            # Kernel pour détecter les formes géométriques
            kernel = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.uint8)
            
            gray = cv2.cvtColor(optimized, cv2.COLOR_BGR2GRAY)
            enhanced_symbols = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Combiner avec l'image originale
            enhanced_symbols = cv2.cvtColor(enhanced_symbols, cv2.COLOR_GRAY2BGR)
            optimized = cv2.addWeighted(optimized, 0.7, enhanced_symbols, 0.3, 0)
            
            # 4. RÉDUCTION DU BRUIT SPÉCIFIQUE AUX CARTES
            # Filtre médian pour préserver les bords nets des cartes
            optimized = cv2.medianBlur(optimized, 3)
            
            return optimized
            
        except Exception as e:
            self.logger.error(f"Erreur optimisation cartes: {e}")
            return image
    
    def cleanup(self):
        """Nettoie les ressources"""
        try:
            if self.mss_instance:
                self.mss_instance.close()
                self.logger.info("Ressources MSS libérées")
        except Exception as e:
            self.logger.error(f"Erreur nettoyage: {e}") 