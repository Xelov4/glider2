#!/usr/bin/env python3
"""
Test détaillé de reconnaissance des cartes avec debug
Analyse approfondie de la détection main + flop
"""

import time
import cv2
import numpy as np
from modules.screen_capture import ScreenCapture
from modules.image_analysis import ImageAnalyzer
import json
from datetime import datetime
import os

class DetailedCardRecognitionTester:
    """Testeur détaillé de reconnaissance de cartes"""
    
    def __init__(self):
        self.screen_capture = ScreenCapture()
        self.image_analyzer = ImageAnalyzer()
        self.test_duration = 600  # 10 minutes
        self.capture_interval = 3  # 3 secondes entre captures
        
        # Créer dossier de debug
        self.debug_dir = "debug_cards"
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
        
        # Statistiques détaillées
        self.stats = {
            'total_captures': 0,
            'hand_detections': 0,
            'flop_detections': 0,
            'hand_cards_with_suits': 0,
            'flop_cards_with_suits': 0,
            'hand_cards_without_suits': 0,
            'flop_cards_without_suits': 0,
            'hand_cards_list': [],
            'flop_cards_list': [],
            'detection_times': [],
            'errors': 0
        }
        
        # Cache pour éviter les répétitions
        self.last_hand = []
        self.last_flop = []
        self.capture_count = 0
        
    def run_detailed_test(self):
        """Lance le test détaillé"""
        print("🔍 TEST DÉTAILLÉ DE RECONNAISSANCE DE CARTES")
        print("=" * 70)
        print(f"⏱️  Durée: {self.test_duration} secondes")
        print(f"📸 Intervalle: {self.capture_interval} secondes")
        print(f"📁 Debug: {self.debug_dir}/")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < self.test_duration:
                self.capture_count += 1
                current_time = datetime.now().strftime("%H:%M:%S")
                
                print(f"\n📸 CAPTURE #{self.capture_count} - {current_time}")
                print("-" * 50)
                
                # Capture et analyse
                self.capture_and_analyze(current_time)
                
                # Attente
                time.sleep(self.capture_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Test interrompu par l'utilisateur")
        
        # Statistiques finales
        self.display_detailed_stats()
        
    def capture_and_analyze(self, timestamp):
        """Capture et analyse les cartes"""
        try:
            # Capture des zones
            hand_img = self.screen_capture.capture_region('hand_area')
            flop_img = self.screen_capture.capture_region('community_cards')
            
            if hand_img is None or flop_img is None:
                print("   ⚠️  Impossible de capturer les zones")
                self.stats['errors'] += 1
                return
            
            # Sauvegarde des images de debug
            hand_debug_path = os.path.join(self.debug_dir, f"hand_{self.capture_count:03d}.png")
            flop_debug_path = os.path.join(self.debug_dir, f"flop_{self.capture_count:03d}.png")
            cv2.imwrite(hand_debug_path, hand_img)
            cv2.imwrite(flop_debug_path, flop_img)
            
            print(f"   💾 Images sauvegardées: hand_{self.capture_count:03d}.png, flop_{self.capture_count:03d}.png")
            
            # Analyse détaillée de la main
            print("   🃏 ANALYSE MAIN JOUEUR:")
            hand_cards = self.analyze_cards_detailed(hand_img, "main")
            
            # Analyse détaillée du flop
            print("   🃏 ANALYSE FLOP:")
            flop_cards = self.analyze_cards_detailed(flop_img, "flop")
            
            # Mise à jour des statistiques
            self.update_detailed_stats(hand_cards, flop_cards, timestamp)
            
            # Affichage des résultats
            self.display_detailed_results(hand_cards, flop_cards)
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            self.stats['errors'] += 1
    
    def analyze_cards_detailed(self, img, card_type):
        """Analyse détaillée des cartes"""
        try:
            # Détection des cartes
            cards = self.image_analyzer.detect_cards(img)
            
            if not cards:
                print(f"      ⚠️  Aucune carte détectée")
                return []
            
            # Analyse détaillée
            card_details = []
            cards_with_suits = 0
            cards_without_suits = 0
            
            for card in cards:
                card_str = f"{card.rank}{card.suit}"
                card_details.append(card_str)
                
                if card.suit != '?':
                    cards_with_suits += 1
                else:
                    cards_without_suits += 1
            
            print(f"      📊 {len(cards)} cartes détectées:")
            print(f"         ✅ Avec suit: {cards_with_suits}")
            print(f"         ⚠️  Sans suit: {cards_without_suits}")
            print(f"         🎯 Cartes: {card_details}")
            
            # Validation selon le type
            if card_type == "main":
                if len(cards) > 2:
                    print(f"         ⚠️  ATTENTION: Main avec {len(cards)} cartes (normalement 2)")
                elif len(cards) == 2:
                    print(f"         ✅ Main valide: {len(cards)} cartes")
            elif card_type == "flop":
                if len(cards) > 5:
                    print(f"         ⚠️  ATTENTION: Flop avec {len(cards)} cartes (normalement 3-5)")
                elif 3 <= len(cards) <= 5:
                    print(f"         ✅ Flop valide: {len(cards)} cartes")
                else:
                    print(f"         ⚠️  Flop suspect: {len(cards)} cartes")
            
            return {
                'cards': card_details,
                'with_suits': cards_with_suits,
                'without_suits': cards_without_suits,
                'total': len(cards)
            }
            
        except Exception as e:
            print(f"      ❌ Erreur analyse {card_type}: {e}")
            return []
    
    def update_detailed_stats(self, hand_result, flop_result, timestamp):
        """Met à jour les statistiques détaillées"""
        self.stats['total_captures'] += 1
        
        # Main
        if hand_result and hand_result['cards']:
            self.stats['hand_detections'] += 1
            self.stats['hand_cards_with_suits'] += hand_result['with_suits']
            self.stats['hand_cards_without_suits'] += hand_result['without_suits']
            
            if hand_result['cards'] != self.last_hand:
                self.stats['hand_cards_list'].append({
                    'time': timestamp,
                    'cards': hand_result['cards'],
                    'with_suits': hand_result['with_suits'],
                    'without_suits': hand_result['without_suits']
                })
                self.last_hand = hand_result['cards']
        
        # Flop
        if flop_result and flop_result['cards']:
            self.stats['flop_detections'] += 1
            self.stats['flop_cards_with_suits'] += flop_result['with_suits']
            self.stats['flop_cards_without_suits'] += flop_result['without_suits']
            
            if flop_result['cards'] != self.last_flop:
                self.stats['flop_cards_list'].append({
                    'time': timestamp,
                    'cards': flop_result['cards'],
                    'with_suits': flop_result['with_suits'],
                    'without_suits': flop_result['without_suits']
                })
                self.last_flop = flop_result['cards']
    
    def display_detailed_results(self, hand_result, flop_result):
        """Affiche les résultats détaillés"""
        print(f"   📊 RÉSULTATS DÉTAILLÉS:")
        
        if hand_result:
            print(f"      Main: {hand_result['cards']}")
            print(f"         Suits: {hand_result['with_suits']}/{hand_result['total']}")
        else:
            print(f"      Main: Aucune")
        
        if flop_result:
            print(f"      Flop: {flop_result['cards']}")
            print(f"         Suits: {flop_result['with_suits']}/{flop_result['total']}")
        else:
            print(f"      Flop: Aucune")
    
    def display_detailed_stats(self):
        """Affiche les statistiques détaillées finales"""
        print("\n" + "=" * 70)
        print("📊 STATISTIQUES DÉTAILLÉES FINALES")
        print("=" * 70)
        
        total = self.stats['total_captures']
        if total == 0:
            print("❌ Aucune capture effectuée")
            return
        
        print(f"📸 Captures totales: {total}")
        print(f"🎯 Détections main: {self.stats['hand_detections']} ({self.stats['hand_detections']/total*100:.1f}%)")
        print(f"🎯 Détections flop: {self.stats['flop_detections']} ({self.stats['flop_detections']/total*100:.1f}%)")
        print(f"❌ Erreurs: {self.stats['errors']} ({self.stats['errors']/total*100:.1f}%)")
        
        # Statistiques des suits
        if self.stats['hand_detections'] > 0:
            hand_suit_rate = self.stats['hand_cards_with_suits'] / (self.stats['hand_cards_with_suits'] + self.stats['hand_cards_without_suits']) * 100
            print(f"🎯 Reconnaissance suits main: {hand_suit_rate:.1f}%")
        
        if self.stats['flop_detections'] > 0:
            flop_suit_rate = self.stats['flop_cards_with_suits'] / (self.stats['flop_cards_with_suits'] + self.stats['flop_cards_without_suits']) * 100
            print(f"🎯 Reconnaissance suits flop: {flop_suit_rate:.1f}%")
        
        # Affichage des détections récentes
        if self.stats['hand_cards_list']:
            print(f"\n🃏 MAINS DÉTECTÉES ({len(self.stats['hand_cards_list'])}):")
            for hand in self.stats['hand_cards_list'][-3:]:
                print(f"   {hand['time']}: {hand['cards']} (suits: {hand['with_suits']}/{hand['with_suits']+hand['without_suits']})")
        
        if self.stats['flop_cards_list']:
            print(f"\n🃏 FLOP DÉTECTÉS ({len(self.stats['flop_cards_list'])}):")
            for flop in self.stats['flop_cards_list'][-3:]:
                print(f"   {flop['time']}: {flop['cards']} (suits: {flop['with_suits']}/{flop['with_suits']+flop['without_suits']})")
        
        print(f"\n📁 Images de debug sauvegardées dans: {self.debug_dir}/")
        print("🎯 CONCLUSION:")
        
        if self.stats['hand_detections'] > 0 or self.stats['flop_detections'] > 0:
            print("✅ Reconnaissance de cartes fonctionnelle")
            if hand_suit_rate < 50 or flop_suit_rate < 50:
                print("⚠️  Problème de reconnaissance des suits détecté")
            else:
                print("✅ Reconnaissance des suits satisfaisante")
        else:
            print("⚠️  Aucune carte détectée - vérifiez la fenêtre poker")
        
        print("✅ Test terminé")

def main():
    """Fonction principale"""
    print("🔍 TEST DÉTAILLÉ DE RECONNAISSANCE DE CARTES")
    print("Ce test va analyser en profondeur la détection des cartes")
    print("Les images seront sauvegardées pour debug")
    print("Assurez-vous que votre fenêtre poker est ouverte")
    print("Appuyez sur Ctrl+C pour arrêter le test")
    print()
    
    tester = DetailedCardRecognitionTester()
    tester.run_detailed_test()

if __name__ == "__main__":
    main() 