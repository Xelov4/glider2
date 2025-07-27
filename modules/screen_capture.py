"""
Module de capture d'écran optimisé pour l'agent IA Poker
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
from .constants import DEFAULT_REGIONS, Position, Action, GamePhase

@dataclass
class ScreenRegion:
    """Représente une région de l'écran à capturer"""
    x: int
    y: int
    width: int
    height: int
    name: str
    enabled: bool = True

class ScreenCapture:
    """
    Module de capture d'écran optimisé avec calibration
    """
    
    def __init__(self, config_file: str = "config.ini"):
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        self.regions = {}
        self.calibrated_regions_file = "calibrated_regions.json"
        
        # Charger les régions calibrées ou utiliser les défauts
        self.load_regions()
        
        # Configuration de capture
        self.capture_fps = 10
        self.debug_mode = False
        self.target_window_title = "Betclic Poker"
        
        # Cache pour optimiser les performances
        self.last_capture_time = 0
        self.capture_cache = {}
        self.cache_duration = 0.1  # 100ms
        
    def load_regions(self):
        """Charge les régions depuis le fichier calibré ou utilise les défauts"""
        try:
            if os.path.exists(self.calibrated_regions_file):
                with open(self.calibrated_regions_file, 'r', encoding='utf-8') as f:
                    calibrated_data = json.load(f)
                    
                # Convertir les données calibrées en ScreenRegion
                for region_name, region_data in calibrated_data.items():
                    self.regions[region_name] = ScreenRegion(
                        x=region_data['x'],
                        y=region_data['y'],
                        width=region_data['width'],
                        height=region_data['height'],
                        name=region_data.get('name', region_name),
                        enabled=region_data.get('enabled', True)
                    )
                    
                self.logger.info(f"Régions calibrées chargées: {len(self.regions)} régions")
                
            else:
                # Utiliser les régions par défaut
                for region_name, region_data in DEFAULT_REGIONS.items():
                    self.regions[region_name] = ScreenRegion(
                        x=region_data['x'],
                        y=region_data['y'],
                        width=region_data['width'],
                        height=region_data['height'],
                        name=region_data['name'],
                        enabled=True
                    )
                    
                self.logger.info(f"Régions par défaut chargées: {len(self.regions)} régions")
                
        except Exception as e:
            self.logger.error(f"Erreur chargement régions: {e}")
            # Fallback sur les régions par défaut
            for region_name, region_data in DEFAULT_REGIONS.items():
                self.regions[region_name] = ScreenRegion(
                    x=region_data['x'],
                    y=region_data['y'],
                    width=region_data['width'],
                    height=region_data['height'],
                    name=region_data['name'],
                    enabled=True
                )
    
    def capture_region(self, region_name: str) -> Optional[np.ndarray]:
        """Capture une région spécifique de l'écran en haute qualité"""
        try:
            if region_name not in self.regions:
                self.logger.warning(f"Région inconnue: {region_name}")
                return None
                
            region = self.regions[region_name]
            if not region.enabled:
                return None
                
            # Vérifier le cache
            current_time = time.time()
            if (region_name in self.capture_cache and 
                current_time - self.last_capture_time < self.cache_duration):
                return self.capture_cache[region_name]
            
            # NOUVEAU: Système hybride de capture maximale + post-traitement
            try:
                from .hybrid_capture_system import HybridCaptureSystem, CaptureConfig
                
                # Configuration optimisée pour les cartes
                config = CaptureConfig(
                    capture_method="max_quality",
                    post_processing=True,
                    upscale_factor=3.0,  # Upscaling agressif
                    enhancement_strength=1.8,
                    noise_reduction=True,
                    sharpening=True,
                    contrast_boost=True
                )
                
                hybrid_capture = HybridCaptureSystem(config)
                image = hybrid_capture.capture_with_max_quality(region_name)
                
                if image is not None:
                    # Métriques de performance
                    metrics = hybrid_capture.get_performance_metrics()
                    self.logger.debug(f"Capture hybride {region_name}: amélioration +{metrics['avg_quality_improvement']:.1f}%")
                    
                    # Mettre en cache
                    self.capture_cache[region_name] = image
                    self.last_capture_time = current_time
                    
                    return image
                    
            except Exception as e:
                self.logger.debug(f"Capture hybride non disponible: {e}")
            
            # FALLBACK: Capture standard si HQ échoue
            screenshot = pyautogui.screenshot(region=(region.x, region.y, region.width, region.height))
            image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # SYSTÈME HYBRIDE: Amélioration intelligente pour les cartes
            if region_name in ['hand_area', 'community_cards']:
                # Upscaling agressif pour les cartes
                height, width = image.shape[:2]
                if width < 600:  # Seuil plus élevé pour plus de qualité
                    scale_factor = 3.0  # Facteur plus agressif
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    
                    # Amélioration avancée du contraste
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    lab[:,:,0] = clahe.apply(lab[:,:,0])
                    
                    # Amélioration des couleurs
                    lab[:,:,1] = cv2.multiply(lab[:,:,1], 1.2)
                    lab[:,:,2] = cv2.multiply(lab[:,:,2], 1.2)
                    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    
                    # Sharpening pour les symboles de cartes
                    kernel = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                     [-1,-1,-1]]) * 1.5
                    sharpened = cv2.filter2D(image, -1, kernel)
                    image = cv2.addWeighted(image, 0.6, sharpened, 0.4, 0)
                    
                    self.logger.debug(f"Image hybride améliorée {region_name}: {width}x{height} → {new_width}x{new_height}")
            
            # Mettre en cache
            self.capture_cache[region_name] = image
            self.last_capture_time = current_time
            
            return image
            
        except Exception as e:
            self.logger.error(f"Erreur capture région {region_name}: {e}")
            return None
    
    def capture_all_regions(self) -> Dict[str, np.ndarray]:
        """Capture toutes les régions activées"""
        captured_regions = {}
        
        try:
            for region_name, region in self.regions.items():
                if region.enabled:
                    image = self.capture_region(region_name)
                    if image is not None:
                        captured_regions[region_name] = image
                        
        except Exception as e:
            self.logger.error(f"Erreur capture toutes régions: {e}")
            
        return captured_regions
    
    def get_region_coordinates(self, region_name: str) -> Optional[Tuple[int, int, int, int]]:
        """Retourne les coordonnées d'une région (x, y, width, height)"""
        try:
            if region_name in self.regions:
                region = self.regions[region_name]
                return (region.x, region.y, region.width, region.height)
            else:
                self.logger.warning(f"Région '{region_name}' non trouvée")
                return None
        except Exception as e:
            self.logger.error(f"Erreur récupération coordonnées {region_name}: {e}")
            return None

    def get_region_info(self, region_name: str) -> Optional[Dict]:
        """Retourne les informations d'une région"""
        if region_name in self.regions:
            region = self.regions[region_name]
            return {
                'x': region.x,
                'y': region.y,
                'width': region.width,
                'height': region.height,
                'name': region.name,
                'enabled': region.enabled
            }
        return None
    
    def enable_region(self, region_name: str, enabled: bool = True):
        """Active/désactive une région"""
        if region_name in self.regions:
            self.regions[region_name].enabled = enabled
            self.logger.info(f"Région {region_name} {'activée' if enabled else 'désactivée'}")
    
    def update_region_coordinates(self, region_name: str, x: int, y: int, width: int, height: int):
        """Met à jour les coordonnées d'une région"""
        if region_name in self.regions:
            self.regions[region_name].x = x
            self.regions[region_name].y = y
            self.regions[region_name].width = width
            self.regions[region_name].height = height
            self.logger.info(f"Région {region_name} mise à jour: ({x}, {y}, {width}, {height})")
    
    def save_regions_to_file(self, filename: str = None):
        """Sauvegarde les régions dans un fichier JSON"""
        if filename is None:
            filename = self.calibrated_regions_file
            
        try:
            regions_data = {}
            for region_name, region in self.regions.items():
                regions_data[region_name] = {
                    'x': region.x,
                    'y': region.y,
                    'width': region.width,
                    'height': region.height,
                    'name': region.name,
                    'enabled': region.enabled
                }
                
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(regions_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Régions sauvegardées dans {filename}")
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde régions: {e}")
    
    def validate_regions(self) -> List[str]:
        """Valide que toutes les régions sont dans les limites de l'écran"""
        errors = []
        screen_width, screen_height = pyautogui.size()
        
        for region_name, region in self.regions.items():
            if region.x < 0 or region.y < 0:
                errors.append(f"Région {region_name}: coordonnées négatives")
            elif region.x + region.width > screen_width:
                errors.append(f"Région {region_name}: dépasse la largeur de l'écran")
            elif region.y + region.height > screen_height:
                errors.append(f"Région {region_name}: dépasse la hauteur de l'écran")
                
        return errors
    
    def get_region_statistics(self) -> Dict:
        """Retourne des statistiques sur les régions"""
        total_regions = len(self.regions)
        enabled_regions = sum(1 for r in self.regions.values() if r.enabled)
        
        # Calculer la surface totale capturée
        total_area = sum(r.width * r.height for r in self.regions.values() if r.enabled)
        
        return {
            'total_regions': total_regions,
            'enabled_regions': enabled_regions,
            'disabled_regions': total_regions - enabled_regions,
            'total_capture_area': total_area,
            'regions_list': list(self.regions.keys())
        }
    
    def capture_full_screen(self) -> Optional[np.ndarray]:
        """Capture l'écran complet pour détection de la couronne de victoire"""
        try:
            # Capture de l'écran complet
            screenshot = pyautogui.screenshot()
            if screenshot is None:
                self.logger.error("Impossible de capturer l'écran complet")
                return None
            
            # Convertir en format OpenCV
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            self.logger.debug(f"Écran complet capturé: {screenshot_cv.shape}")
            return screenshot_cv
            
        except Exception as e:
            self.logger.error(f"Erreur capture écran complet: {e}")
            return None 