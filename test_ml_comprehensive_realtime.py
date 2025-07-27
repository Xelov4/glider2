#!/usr/bin/env python3
"""
🧠 TEST ULTRA-COMPLET ML EN TEMPS RÉEL
=======================================

Test complet du système ML pour analyser toutes les cartes en temps réel :
- Votre main (2 cartes)
- Flop (3 cartes)
- Turn (1 carte)
- River (1 carte)
- Jetons et montants
- Métriques de performance avancées
"""

import sys
import os
import time
import cv2
import numpy as np
import logging
from datetime import datetime
import threading
from collections import defaultdict

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RealTimeCardAnalyzer:
    """
    Analyseur de cartes en temps réel avec métriques avancées
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Import des modules
        from modules.screen_capture import ScreenCapture
        from modules.image_analysis import ImageAnalyzer
        from modules.card_ml_detector import CardMLDetector
        
        self.screen_capture = ScreenCapture()
        self.image_analyzer = ImageAnalyzer()
        self.ml_detector = CardMLDetector()
        
        # Métriques de performance
        self.performance_metrics = {
            'total_cycles': 0,
            'detection_times': [],
            'cards_detected': defaultdict(int),
            'confidence_scores': [],
            'error_count': 0,
            'success_rate': 0.0,
            'avg_detection_time': 0.0,
            'ml_vs_ocr_comparison': {'ml_wins': 0, 'ocr_wins': 0, 'ties': 0}
        }
        
        # Historique des détections
        self.detection_history = {
            'hand_cards': [],
            'flop_cards': [],
            'turn_cards': [],
            'river_cards': [],
            'pot_amounts': [],
            'stack_amounts': []
        }
        
        # Configuration
        self.test_duration = 300  # 5 minutes
        self.capture_interval = 2.0  # 2 secondes entre captures
        self.running = False
        
    def start_comprehensive_test(self):
        """Lance le test ultra-complet en temps réel"""
        print("🧠 TEST ULTRA-COMPLET ML EN TEMPS RÉEL")
        print("=" * 60)
        print(f"⏰ Début: {datetime.now().strftime('%H:%M:%S')}")
        print(f"⏱️  Durée: {self.test_duration} secondes")
        print(f"📊 Intervalle: {self.capture_interval}s")
        print()
        
        print("🎯 RÉGIONS ANALYSÉES:")
        print("  📋 hand_area - Votre main (2 cartes)")
        print("  🃏 community_cards - Flop/Turn/River")
        print("  💰 pot_area - Montant du pot")
        print("  🪙 my_stack_area - Vos jetons")
        print("  🎮 fold_button, call_button, raise_button")
        print()
        
        print("📈 MÉTRIQUES COLLECTÉES:")
        print("  ⚡ Temps de détection")
        print("  🎯 Précision ML vs OCR")
        print("  🃏 Cartes détectées par région")
        print("  📊 Scores de confiance")
        print("  🔄 Historique des mains")
        print()
        
        print("🚀 DÉMARRAGE DU TEST...")
        print("💡 Jouez normalement, le système analyse en arrière-plan!")
        print("-" * 60)
        
        self.running = True
        start_time = time.time()
        
        try:
            while self.running and (time.time() - start_time) < self.test_duration:
                cycle_start = time.time()
                
                # Analyse complète de toutes les régions
                self._analyze_all_regions()
                
                # Affichage des résultats en temps réel
                self._display_realtime_results()
                
                # Calcul des métriques
                self._update_performance_metrics()
                
                # Attendre avant la prochaine capture
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.capture_interval - cycle_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                self.performance_metrics['total_cycles'] += 1
                
        except KeyboardInterrupt:
            print("\n⏹️  Test interrompu par l'utilisateur")
        except Exception as e:
            print(f"\n❌ Erreur pendant le test: {e}")
        finally:
            self.running = False
            self._display_final_results()
    
    def _analyze_all_regions(self):
        """Analyse toutes les régions en une fois"""
        try:
            regions_to_analyze = [
                'hand_area',
                'community_cards', 
                'pot_area',
                'my_stack_area',
                'fold_button',
                'call_button',
                'raise_button'
            ]
            
            for region_name in regions_to_analyze:
                self._analyze_single_region(region_name)
                
        except Exception as e:
            self.logger.error(f"Erreur analyse régions: {e}")
            self.performance_metrics['error_count'] += 1
    
    def _analyze_single_region(self, region_name: str):
        """Analyse une région spécifique avec ML et OCR"""
        try:
            # Capture de la région
            region_image = self.screen_capture.capture_region(region_name)
            if region_image is None:
                return
            
            detection_start = time.time()
            
            # 1. DÉTECTION ML (priorité)
            ml_cards = []
            ml_time = 0
            try:
                ml_start = time.time()
                ml_cards = self.ml_detector.detect_cards_ml(region_image)
                ml_time = time.time() - ml_start
            except Exception as e:
                self.logger.debug(f"ML échoué pour {region_name}: {e}")
            
            # 2. DÉTECTION OCR (comparaison)
            ocr_cards = []
            ocr_time = 0
            try:
                ocr_start = time.time()
                ocr_cards = self.image_analyzer._detect_cards_ocr_optimized(region_image)
                ocr_time = time.time() - ocr_start
            except Exception as e:
                self.logger.debug(f"OCR échoué pour {region_name}: {e}")
            
            # 3. ANALYSE DE COULEUR
            color_analysis = {}
            try:
                color_analysis = self.ml_detector._analyze_color_ml(region_image)
            except Exception as e:
                self.logger.debug(f"Analyse couleur échouée: {e}")
            
            # 4. EXTRACTION DE TEXTE (pour montants)
            text = ""
            try:
                text = self.image_analyzer.extract_text(region_image, region_name)
            except Exception as e:
                self.logger.debug(f"Extraction texte échouée: {e}")
            
            detection_time = time.time() - detection_start
            
            # Stocker les résultats
            result = {
                'region': region_name,
                'ml_cards': ml_cards,
                'ocr_cards': ocr_cards,
                'ml_time': ml_time,
                'ocr_time': ocr_time,
                'total_time': detection_time,
                'color_analysis': color_analysis,
                'text': text,
                'timestamp': datetime.now()
            }
            
            # Mettre à jour l'historique selon la région
            self._update_detection_history(region_name, result)
            
            # Métriques de performance
            self.performance_metrics['detection_times'].append(detection_time)
            
            # Comparaison ML vs OCR
            if ml_cards and not ocr_cards:
                self.performance_metrics['ml_vs_ocr_comparison']['ml_wins'] += 1
            elif ocr_cards and not ml_cards:
                self.performance_metrics['ml_vs_ocr_comparison']['ocr_wins'] += 1
            elif ml_cards and ocr_cards:
                self.performance_metrics['ml_vs_ocr_comparison']['ties'] += 1
            
            # Compter les cartes détectées
            for card in ml_cards:
                card_key = f"{card.rank}{card.suit}"
                self.performance_metrics['cards_detected'][card_key] += 1
                self.performance_metrics['confidence_scores'].append(card.confidence)
            
        except Exception as e:
            self.logger.error(f"Erreur analyse {region_name}: {e}")
            self.performance_metrics['error_count'] += 1
    
    def _update_detection_history(self, region_name: str, result: dict):
        """Met à jour l'historique des détections"""
        try:
            if region_name == 'hand_area':
                self.detection_history['hand_cards'].append({
                    'cards': [f"{c.rank}{c.suit}" for c in result['ml_cards']],
                    'confidence': [c.confidence for c in result['ml_cards']],
                    'timestamp': result['timestamp']
                })
            elif region_name == 'community_cards':
                self.detection_history['flop_cards'].append({
                    'cards': [f"{c.rank}{c.suit}" for c in result['ml_cards']],
                    'confidence': [c.confidence for c in result['ml_cards']],
                    'timestamp': result['timestamp']
                })
            elif region_name == 'pot_area':
                # Extraire le montant du pot
                import re
                numbers = re.findall(r'\d+', result['text'])
                if numbers:
                    self.detection_history['pot_amounts'].append({
                        'amount': int(numbers[0]),
                        'timestamp': result['timestamp']
                    })
            elif region_name == 'my_stack_area':
                # Extraire le montant de la stack
                import re
                numbers = re.findall(r'\d+', result['text'])
                if numbers:
                    self.detection_history['stack_amounts'].append({
                        'amount': int(numbers[0]),
                        'timestamp': result['timestamp']
                    })
                    
        except Exception as e:
            self.logger.debug(f"Erreur mise à jour historique: {e}")
    
    def _display_realtime_results(self):
        """Affiche les résultats en temps réel"""
        try:
            # Obtenir les dernières détections
            current_hand = self.detection_history['hand_cards'][-1] if self.detection_history['hand_cards'] else None
            current_flop = self.detection_history['flop_cards'][-1] if self.detection_history['flop_cards'] else None
            current_pot = self.detection_history['pot_amounts'][-1] if self.detection_history['pot_amounts'] else None
            current_stack = self.detection_history['stack_amounts'][-1] if self.detection_history['stack_amounts'] else None
            
            # Calculer les métriques actuelles
            total_cycles = self.performance_metrics['total_cycles']
            avg_time = np.mean(self.performance_metrics['detection_times']) if self.performance_metrics['detection_times'] else 0
            ml_wins = self.performance_metrics['ml_vs_ocr_comparison']['ml_wins']
            ocr_wins = self.performance_metrics['ml_vs_ocr_comparison']['ocr_wins']
            
            # Affichage en temps réel
            print(f"\r🔄 Cycle {total_cycles:3d} | ⏱️  {avg_time:.3f}s | 🧠 ML: {ml_wins} | 🔍 OCR: {ocr_wins} | ", end="")
            
            if current_hand:
                hand_str = " ".join(current_hand['cards'])
                print(f"📋 Main: {hand_str} | ", end="")
            
            if current_flop:
                flop_str = " ".join(current_flop['cards'])
                print(f"🃏 Flop: {flop_str} | ", end="")
            
            if current_pot:
                print(f"💰 Pot: {current_pot['amount']} | ", end="")
            
            if current_stack:
                print(f"🪙 Stack: {current_stack['amount']}", end="")
            
            print(" " * 20, end="\r")  # Nettoyer la ligne
            
        except Exception as e:
            self.logger.debug(f"Erreur affichage temps réel: {e}")
    
    def _update_performance_metrics(self):
        """Met à jour les métriques de performance"""
        try:
            if self.performance_metrics['detection_times']:
                self.performance_metrics['avg_detection_time'] = np.mean(self.performance_metrics['detection_times'])
            
            total_attempts = self.performance_metrics['total_cycles']
            total_errors = self.performance_metrics['error_count']
            if total_attempts > 0:
                self.performance_metrics['success_rate'] = (total_attempts - total_errors) / total_attempts * 100
                
        except Exception as e:
            self.logger.debug(f"Erreur mise à jour métriques: {e}")
    
    def _display_final_results(self):
        """Affiche les résultats finaux complets"""
        print("\n" + "=" * 60)
        print("📊 RÉSULTATS FINAUX - TEST ULTRA-COMPLET ML")
        print("=" * 60)
        
        # Métriques de performance
        print(f"⏱️  TEMPS TOTAL: {self.test_duration} secondes")
        print(f"🔄 CYCLES TOTAUX: {self.performance_metrics['total_cycles']}")
        print(f"⚡ TEMPS MOYEN: {self.performance_metrics['avg_detection_time']:.3f}s")
        print(f"✅ TAUX DE RÉUSSITE: {self.performance_metrics['success_rate']:.1f}%")
        print(f"❌ ERREURS: {self.performance_metrics['error_count']}")
        
        # Comparaison ML vs OCR
        ml_wins = self.performance_metrics['ml_vs_ocr_comparison']['ml_wins']
        ocr_wins = self.performance_metrics['ml_vs_ocr_comparison']['ocr_wins']
        ties = self.performance_metrics['ml_vs_ocr_comparison']['ties']
        print(f"\n🧠 COMPARAISON ML vs OCR:")
        print(f"  🧠 ML gagne: {ml_wins}")
        print(f"  🔍 OCR gagne: {ocr_wins}")
        print(f"  🤝 Égalités: {ties}")
        
        # Cartes détectées
        print(f"\n🃏 CARTES DÉTECTÉES:")
        for card, count in sorted(self.performance_metrics['cards_detected'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {card}: {count} fois")
        
        # Scores de confiance
        if self.performance_metrics['confidence_scores']:
            avg_confidence = np.mean(self.performance_metrics['confidence_scores'])
            min_confidence = min(self.performance_metrics['confidence_scores'])
            max_confidence = max(self.performance_metrics['confidence_scores'])
            print(f"\n📊 SCORES DE CONFIANCE:")
            print(f"  📈 Moyenne: {avg_confidence:.3f}")
            print(f"  📉 Minimum: {min_confidence:.3f}")
            print(f"  📈 Maximum: {max_confidence:.3f}")
        
        # Historique des mains
        print(f"\n📋 HISTORIQUE DES MAINS:")
        for i, hand in enumerate(self.detection_history['hand_cards'][-10:], 1):  # 10 dernières mains
            cards_str = " ".join(hand['cards'])
            conf_str = f"[{', '.join(f'{c:.2f}' for c in hand['confidence'])}]"
            time_str = hand['timestamp'].strftime('%H:%M:%S')
            print(f"  {i:2d}. {cards_str} {conf_str} ({time_str})")
        
        # Historique du flop
        print(f"\n🃏 HISTORIQUE DU FLOP:")
        for i, flop in enumerate(self.detection_history['flop_cards'][-10:], 1):  # 10 derniers flops
            cards_str = " ".join(flop['cards'])
            conf_str = f"[{', '.join(f'{c:.2f}' for c in flop['confidence'])}]"
            time_str = flop['timestamp'].strftime('%H:%M:%S')
            print(f"  {i:2d}. {cards_str} {conf_str} ({time_str})")
        
        # Montants
        if self.detection_history['pot_amounts']:
            print(f"\n💰 MONTANTS DU POT:")
            for amount_data in self.detection_history['pot_amounts'][-5:]:  # 5 derniers
                time_str = amount_data['timestamp'].strftime('%H:%M:%S')
                print(f"  {amount_data['amount']} jetons ({time_str})")
        
        if self.detection_history['stack_amounts']:
            print(f"\n🪙 MONTANTS DE LA STACK:")
            for amount_data in self.detection_history['stack_amounts'][-5:]:  # 5 derniers
                time_str = amount_data['timestamp'].strftime('%H:%M:%S')
                print(f"  {amount_data['amount']} jetons ({time_str})")
        
        print(f"\n⏰ Fin du test: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)

def main():
    """Fonction principale"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    analyzer = RealTimeCardAnalyzer()
    analyzer.start_comprehensive_test()

if __name__ == "__main__":
    main() 