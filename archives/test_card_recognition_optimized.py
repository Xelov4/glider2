#!/usr/bin/env python3
"""
Test optimisé de reconnaissance des cartes
Se concentre sur la qualité de détection avec seuils stricts
"""

import time
import cv2
import numpy as np
from modules.screen_capture import ScreenCapture
from modules.image_analysis import ImageAnalyzer
import json
from datetime import datetime
import os

class OptimizedCardRecognitionTester:
    """Testeur optimisé de reconnaissance de cartes"""
    
    def __init__(self):
        self.screen_capture = ScreenCapture()
        self.image_analyzer = ImageAnalyzer()
        self.test_duration = 300  # 5 minutes
        self.capture_interval = 5  # 5 secondes entre captures
        
        # Statistiques optimisées
        self.stats = {
            'total_captures': 0,
            'valid_hands': 0,  # Mains avec exactement 2 cartes
            'valid_flops': 0,  # Flops avec 3-5 cartes
            'hands_with_suits': 0,
            'flops_with_suits': 0,
            'best_detections': [],
            'errors': 0
        }
        
        # Cache pour éviter les répétitions
        self.last_hand = []
        self.last_flop = []
        
    def run_optimized_test(self):
        """Lance le test optimisé"""
        print("🎯 TEST OPTIMISÉ DE RECONNAISSANCE DE CARTES")
        print("=" * 60)
        print(f"⏱️  Durée: {self.test_duration} secondes")
        print(f"📸 Intervalle: {self.capture_interval} secondes")
        print("🎯 Objectif: Qualité > Quantité")
        print("=" * 60)
        
        start_time = time.time()
        capture_count = 0
        
        try:
            while time.time() - start_time < self.test_duration:
                capture_count += 1
                current_time = datetime.now().strftime("%H:%M:%S")
                
                print(f"\n📸 CAPTURE #{capture_count} - {current_time}")
                print("-" * 40)
                
                # Capture et analyse optimisée
                self.capture_and_analyze_optimized(current_time)
                
                # Attente
                time.sleep(self.capture_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Test interrompu par l'utilisateur")
        
        # Statistiques finales
        self.display_optimized_stats()
        
    def capture_and_analyze_optimized(self, timestamp):
        """Capture et analyse optimisée"""
        try:
            # Capture des zones
            hand_img = self.screen_capture.capture_region('hand_area')
            flop_img = self.screen_capture.capture_region('community_cards')
            
            if hand_img is None or flop_img is None:
                print("   ⚠️  Impossible de capturer les zones")
                self.stats['errors'] += 1
                return
            
            # Analyse optimisée de la main
            print("   🃏 ANALYSE MAIN JOUEUR:")
            hand_result = self.analyze_hand_optimized(hand_img)
            
            # Analyse optimisée du flop
            print("   🃏 ANALYSE FLOP:")
            flop_result = self.analyze_flop_optimized(flop_img)
            
            # Mise à jour des statistiques
            self.update_optimized_stats(hand_result, flop_result, timestamp)
            
            # Affichage des résultats
            self.display_optimized_results(hand_result, flop_result)
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            self.stats['errors'] += 1
    
    def analyze_hand_optimized(self, img):
        """Analyse optimisée de la main (2 cartes max)"""
        try:
            cards = self.image_analyzer.detect_cards(img)
            
            if not cards:
                print("      ⚠️  Aucune carte détectée")
                return {'valid': False, 'cards': [], 'with_suits': 0}
            
            # Filtrer pour ne garder que les 2 meilleures cartes
            if len(cards) > 2:
                print(f"      ⚠️  {len(cards)} cartes détectées, filtrage des 2 meilleures")
                # Trier par confiance (si disponible) ou prendre les 2 premières
                cards = cards[:2]
            
            card_details = []
            cards_with_suits = 0
            
            for card in cards:
                card_str = f"{card.rank}{card.suit}"
                card_details.append(card_str)
                if card.suit != '?':
                    cards_with_suits += 1
            
            is_valid = len(cards) == 2
            print(f"      📊 {len(cards)} cartes: {card_details}")
            print(f"         Suits: {cards_with_suits}/{len(cards)}")
            print(f"         Validité: {'✅' if is_valid else '❌'}")
            
            return {
                'valid': is_valid,
                'cards': card_details,
                'with_suits': cards_with_suits,
                'total': len(cards)
            }
            
        except Exception as e:
            print(f"      ❌ Erreur analyse main: {e}")
            return {'valid': False, 'cards': [], 'with_suits': 0}
    
    def analyze_flop_optimized(self, img):
        """Analyse optimisée du flop (3-5 cartes)"""
        try:
            cards = self.image_analyzer.detect_cards(img)
            
            if not cards:
                print("      ⚠️  Aucune carte détectée")
                return {'valid': False, 'cards': [], 'with_suits': 0}
            
            # Filtrer pour ne garder que les 3-5 meilleures cartes
            if len(cards) > 5:
                print(f"      ⚠️  {len(cards)} cartes détectées, filtrage des 5 meilleures")
                cards = cards[:5]
            elif len(cards) < 3:
                print(f"      ⚠️  {len(cards)} cartes détectées (minimum 3 attendu)")
            
            card_details = []
            cards_with_suits = 0
            
            for card in cards:
                card_str = f"{card.rank}{card.suit}"
                card_details.append(card_str)
                if card.suit != '?':
                    cards_with_suits += 1
            
            is_valid = 3 <= len(cards) <= 5
            print(f"      📊 {len(cards)} cartes: {card_details}")
            print(f"         Suits: {cards_with_suits}/{len(cards)}")
            print(f"         Validité: {'✅' if is_valid else '❌'}")
            
            return {
                'valid': is_valid,
                'cards': card_details,
                'with_suits': cards_with_suits,
                'total': len(cards)
            }
            
        except Exception as e:
            print(f"      ❌ Erreur analyse flop: {e}")
            return {'valid': False, 'cards': [], 'with_suits': 0}
    
    def update_optimized_stats(self, hand_result, flop_result, timestamp):
        """Met à jour les statistiques optimisées"""
        self.stats['total_captures'] += 1
        
        # Main
        if hand_result['valid']:
            self.stats['valid_hands'] += 1
            if hand_result['with_suits'] == hand_result['total']:
                self.stats['hands_with_suits'] += 1
            
            if hand_result['cards'] != self.last_hand:
                self.stats['best_detections'].append({
                    'time': timestamp,
                    'type': 'hand',
                    'cards': hand_result['cards'],
                    'with_suits': hand_result['with_suits']
                })
                self.last_hand = hand_result['cards']
        
        # Flop
        if flop_result['valid']:
            self.stats['valid_flops'] += 1
            if flop_result['with_suits'] == flop_result['total']:
                self.stats['flops_with_suits'] += 1
            
            if flop_result['cards'] != self.last_flop:
                self.stats['best_detections'].append({
                    'time': timestamp,
                    'type': 'flop',
                    'cards': flop_result['cards'],
                    'with_suits': flop_result['with_suits']
                })
                self.last_flop = flop_result['cards']
    
    def display_optimized_results(self, hand_result, flop_result):
        """Affiche les résultats optimisés"""
        print(f"   📊 RÉSULTATS OPTIMISÉS:")
        
        if hand_result['valid']:
            print(f"      ✅ Main valide: {hand_result['cards']}")
        else:
            print(f"      ❌ Main invalide: {hand_result['cards']}")
        
        if flop_result['valid']:
            print(f"      ✅ Flop valide: {flop_result['cards']}")
        else:
            print(f"      ❌ Flop invalide: {flop_result['cards']}")
    
    def display_optimized_stats(self):
        """Affiche les statistiques optimisées finales"""
        print("\n" + "=" * 60)
        print("📊 STATISTIQUES OPTIMISÉES FINALES")
        print("=" * 60)
        
        total = self.stats['total_captures']
        if total == 0:
            print("❌ Aucune capture effectuée")
            return
        
        print(f"📸 Captures totales: {total}")
        print(f"✅ Mains valides: {self.stats['valid_hands']} ({self.stats['valid_hands']/total*100:.1f}%)")
        print(f"✅ Flops valides: {self.stats['valid_flops']} ({self.stats['valid_flops']/total*100:.1f}%)")
        print(f"🎯 Mains avec suits: {self.stats['hands_with_suits']} ({self.stats['hands_with_suits']/max(1,self.stats['valid_hands'])*100:.1f}%)")
        print(f"🎯 Flops avec suits: {self.stats['flops_with_suits']} ({self.stats['flops_with_suits']/max(1,self.stats['valid_flops'])*100:.1f}%)")
        print(f"❌ Erreurs: {self.stats['errors']} ({self.stats['errors']/total*100:.1f}%)")
        
        # Affichage des meilleures détections
        if self.stats['best_detections']:
            print(f"\n🏆 MEILLEURES DÉTECTIONS ({len(self.stats['best_detections'])}):")
            for detection in self.stats['best_detections'][-5:]:
                print(f"   {detection['time']} - {detection['type'].upper()}: {detection['cards']} (suits: {detection['with_suits']})")
        
        print("\n🎯 CONCLUSION:")
        if self.stats['valid_hands'] > 0 or self.stats['valid_flops'] > 0:
            print("✅ Reconnaissance de cartes optimisée fonctionnelle")
            if self.stats['valid_hands']/total > 0.5 and self.stats['valid_flops']/total > 0.5:
                print("✅ Qualité de détection satisfaisante")
            else:
                print("⚠️  Qualité de détection à améliorer")
        else:
            print("⚠️  Aucune détection valide - vérifiez la fenêtre poker")
        
        print("✅ Test terminé")

def main():
    """Fonction principale"""
    print("🎯 TEST OPTIMISÉ DE RECONNAISSANCE DE CARTES")
    print("Ce test se concentre sur la qualité de détection")
    print("Assurez-vous que votre fenêtre poker est ouverte")
    print("Appuyez sur Ctrl+C pour arrêter le test")
    print()
    
    tester = OptimizedCardRecognitionTester()
    tester.run_optimized_test()

if __name__ == "__main__":
    main() 