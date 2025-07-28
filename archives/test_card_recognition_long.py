#!/usr/bin/env python3
"""
Test long de reconnaissance des cartes
Vérifie systématiquement la main du joueur et le flop
"""

import time
import cv2
import numpy as np
from modules.screen_capture import ScreenCapture
from modules.image_analysis import ImageAnalyzer
import json
from datetime import datetime

class CardRecognitionTester:
    """Testeur de reconnaissance de cartes en continu"""
    
    def __init__(self):
        self.screen_capture = ScreenCapture()
        self.image_analyzer = ImageAnalyzer()
        self.test_duration = 300  # 5 minutes
        self.capture_interval = 2  # 2 secondes entre captures
        
        # Statistiques
        self.stats = {
            'total_captures': 0,
            'hand_detections': 0,
            'flop_detections': 0,
            'no_cards_detected': 0,
            'errors': 0,
            'hand_cards': [],
            'flop_cards': [],
            'detection_times': []
        }
        
        # Cache pour éviter les répétitions
        self.last_hand = []
        self.last_flop = []
        
    def run_long_test(self):
        """Lance le test long de reconnaissance"""
        print("🎯 TEST LONG DE RECONNAISSANCE DE CARTES")
        print("=" * 60)
        print(f"⏱️  Durée: {self.test_duration} secondes")
        print(f"📸 Intervalle: {self.capture_interval} secondes")
        print(f"🎯 Objectif: Vérifier main joueur + flop")
        print("=" * 60)
        
        start_time = time.time()
        capture_count = 0
        
        try:
            while time.time() - start_time < self.test_duration:
                capture_count += 1
                current_time = datetime.now().strftime("%H:%M:%S")
                
                print(f"\n📸 CAPTURE #{capture_count} - {current_time}")
                print("-" * 40)
                
                # Capture des zones
                try:
                    hand_img = self.screen_capture.capture_region('hand_area')
                    flop_img = self.screen_capture.capture_region('community_cards')
                    
                    if hand_img is None or flop_img is None:
                        print("   ⚠️  Impossible de capturer les zones (fenêtre fermée?)")
                        self.stats['errors'] += 1
                        time.sleep(self.capture_interval)
                        continue
                    
                    # Analyse de la main du joueur
                    print("   🃏 ANALYSE MAIN JOUEUR:")
                    hand_cards = self.analyze_player_hand(hand_img)
                    
                    # Analyse du flop
                    print("   🃏 ANALYSE FLOP:")
                    flop_cards = self.analyze_flop(flop_img)
                    
                    # Mise à jour des statistiques
                    self.update_stats(hand_cards, flop_cards)
                    
                    # Affichage des résultats
                    self.display_results(hand_cards, flop_cards)
                    
                except Exception as e:
                    print(f"   ❌ Erreur: {e}")
                    self.stats['errors'] += 1
                
                # Attente avant prochaine capture
                time.sleep(self.capture_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Test interrompu par l'utilisateur")
        
        # Affichage des statistiques finales
        self.display_final_stats()
        
    def analyze_player_hand(self, hand_img):
        """Analyse la main du joueur"""
        try:
            # Détection des cartes avec template matching
            cards = self.image_analyzer.detect_cards(hand_img)
            
            if cards:
                card_names = [f"{card.rank}{card.suit}" for card in cards]
                print(f"      ✅ Cartes détectées: {card_names}")
                return card_names
            else:
                print("      ⚠️  Aucune carte détectée")
                return []
                
        except Exception as e:
            print(f"      ❌ Erreur détection main: {e}")
            return []
    
    def analyze_flop(self, flop_img):
        """Analyse le flop (cartes communautaires)"""
        try:
            # Détection des cartes avec template matching
            cards = self.image_analyzer.detect_cards(flop_img)
            
            if cards:
                card_names = [f"{card.rank}{card.suit}" for card in cards]
                print(f"      ✅ Cartes détectées: {card_names}")
                return card_names
            else:
                print("      ⚠️  Aucune carte détectée")
                return []
                
        except Exception as e:
            print(f"      ❌ Erreur détection flop: {e}")
            return []
    
    def update_stats(self, hand_cards, flop_cards):
        """Met à jour les statistiques"""
        self.stats['total_captures'] += 1
        
        if hand_cards:
            self.stats['hand_detections'] += 1
            if hand_cards != self.last_hand:
                self.stats['hand_cards'].append({
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'cards': hand_cards
                })
                self.last_hand = hand_cards
        else:
            self.stats['no_cards_detected'] += 1
        
        if flop_cards:
            self.stats['flop_detections'] += 1
            if flop_cards != self.last_flop:
                self.stats['flop_cards'].append({
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'cards': flop_cards
                })
                self.last_flop = flop_cards
    
    def display_results(self, hand_cards, flop_cards):
        """Affiche les résultats de la capture"""
        print(f"   📊 RÉSULTATS:")
        print(f"      Main: {hand_cards if hand_cards else 'Aucune'}")
        print(f"      Flop: {flop_cards if flop_cards else 'Aucune'}")
        
        # Calcul du temps de détection
        detection_time = time.time()
        self.stats['detection_times'].append(detection_time)
        
        if len(self.stats['detection_times']) > 1:
            avg_time = np.mean(np.diff(self.stats['detection_times'][-10:]))
            print(f"      ⏱️  Temps moyen: {avg_time:.3f}s")
    
    def display_final_stats(self):
        """Affiche les statistiques finales"""
        print("\n" + "=" * 60)
        print("📊 STATISTIQUES FINALES")
        print("=" * 60)
        
        total = self.stats['total_captures']
        if total == 0:
            print("❌ Aucune capture effectuée")
            return
        
        print(f"📸 Captures totales: {total}")
        print(f"🎯 Détections main: {self.stats['hand_detections']} ({self.stats['hand_detections']/total*100:.1f}%)")
        print(f"🎯 Détections flop: {self.stats['flop_detections']} ({self.stats['flop_detections']/total*100:.1f}%)")
        print(f"⚠️  Aucune carte: {self.stats['no_cards_detected']} ({self.stats['no_cards_detected']/total*100:.1f}%)")
        print(f"❌ Erreurs: {self.stats['errors']} ({self.stats['errors']/total*100:.1f}%)")
        
        if self.stats['detection_times']:
            avg_detection_time = np.mean(np.diff(self.stats['detection_times']))
            print(f"⏱️  Temps moyen de détection: {avg_detection_time:.3f}s")
        
        # Affichage des mains détectées
        if self.stats['hand_cards']:
            print(f"\n🃏 MAINS DÉTECTÉES ({len(self.stats['hand_cards'])}):")
            for hand in self.stats['hand_cards'][-5:]:  # 5 dernières
                print(f"   {hand['time']}: {hand['cards']}")
        
        # Affichage des flops détectés
        if self.stats['flop_cards']:
            print(f"\n🃏 FLOP DÉTECTÉS ({len(self.stats['flop_cards'])}):")
            for flop in self.stats['flop_cards'][-5:]:  # 5 derniers
                print(f"   {flop['time']}: {flop['cards']}")
        
        print("\n🎯 CONCLUSION:")
        if self.stats['hand_detections'] > 0 or self.stats['flop_detections'] > 0:
            print("✅ Reconnaissance de cartes fonctionnelle")
        else:
            print("⚠️  Aucune carte détectée - vérifiez la fenêtre poker")
        
        print("✅ Test terminé")

def main():
    """Fonction principale"""
    print("🎯 TEST LONG DE RECONNAISSANCE DE CARTES")
    print("Assurez-vous que votre fenêtre poker est ouverte et visible")
    print("Appuyez sur Ctrl+C pour arrêter le test")
    print()
    
    tester = CardRecognitionTester()
    tester.run_long_test()

if __name__ == "__main__":
    main() 