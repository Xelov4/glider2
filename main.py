"""
Agent IA Poker - Point d'entrée principal
Version unifiée et optimisée
"""

import sys
import time
import logging
import signal
import threading
from typing import Dict, Optional
from pathlib import Path

# Import des modules unifiés
from modules.screen_capture import ScreenCapture
from modules.image_analysis import ImageAnalyzer
from modules.game_state import GameState
from modules.poker_engine import PokerEngine
from modules.ai_decision import AIDecisionMaker
from modules.automation import AutomationEngine
from modules.button_detector import ButtonDetector
from modules.strategy_engine import GeneralStrategy
from modules.spin_rush_strategy import SpinRushStrategy
from modules.constants import Position, Action, GamePhase, DEFAULT_CONFIG

class PokerAgent:
    """
    Agent IA Poker principal - Version unifiée
    """
    
    def __init__(self, config_file: str = "config.ini"):
        self.logger = self._setup_logging()
        self.config_file = config_file
        self.running = False
        self.paused = False
        
        # Initialisation des modules unifiés
        self.screen_capture = ScreenCapture(config_file)
        self.image_analyzer = ImageAnalyzer()
        self.game_state = GameState()
        self.poker_engine = PokerEngine()
        self.button_detector = ButtonDetector()
        
        # Stratégies unifiées
        self.general_strategy = GeneralStrategy()
        self.spin_rush_strategy = SpinRushStrategy()
        self.current_strategy = self.spin_rush_strategy  # Stratégie par défaut: Spin & Rush
        
        # Moteurs de décision et d'automatisation
        self.ai_decision = AIDecisionMaker()
        self.automation = AutomationEngine()
        
        # Configuration
        self.config = self._load_config()
        
        # Statistiques
        self.stats = {
            'hands_played': 0,
            'hands_won': 0,
            'total_profit': 0.0,
            'session_start': time.time()
        }
        
        # Thread de surveillance
        self.monitor_thread = None
        
    def _setup_logging(self) -> logging.Logger:
        """Configure le système de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('poker_ai.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_config(self) -> Dict:
        """Charge la configuration depuis le fichier"""
        try:
            import configparser
            config = configparser.ConfigParser()
            
            if Path(self.config_file).exists():
                config.read(self.config_file)
                self.logger.info(f"Configuration chargée depuis {self.config_file}")
            else:
                # Utiliser la configuration par défaut
                config.read_dict(DEFAULT_CONFIG)
                self.logger.info("Configuration par défaut utilisée")
                
            return config
            
        except Exception as e:
            self.logger.error(f"Erreur chargement config: {e}")
            return DEFAULT_CONFIG
    
    def start(self):
        """Démarre l'agent IA"""
        try:
            self.logger.info("=== DÉMARRAGE DE L'AGENT IA POKER ===")
            
            # Validation des régions
            region_errors = self.screen_capture.validate_regions()
            if region_errors:
                self.logger.warning("Erreurs de validation des régions:")
                for error in region_errors:
                    self.logger.warning(f"  - {error}")
            
            # Statistiques des régions
            stats = self.screen_capture.get_region_statistics()
            self.logger.info(f"Régions configurées: {stats['enabled_regions']}/{stats['total_regions']}")
            
            # Démarrage du thread de surveillance
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            # Boucle principale
            self._main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Arrêt demandé par l'utilisateur")
        except Exception as e:
            self.logger.error(f"Erreur critique: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Arrête l'agent IA"""
        self.logger.info("=== ARRÊT DE L'AGENT IA POKER ===")
        self.running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        
        # Sauvegarde des statistiques
        self._save_session_stats()
        self.logger.info("Agent arrêté proprement")
    
    def pause(self):
        """Met en pause l'agent"""
        self.paused = True
        self.logger.info("Agent mis en pause")
    
    def resume(self):
        """Reprend l'exécution de l'agent"""
        self.paused = False
        self.logger.info("Agent repris")
    
    def _main_loop(self):
        """Boucle principale de l'agent"""
        self.logger.info("Boucle principale démarrée")
        
        while self.running:
            try:
                if self.paused:
                    time.sleep(1)
                    continue
                
                # 1. Capture d'écran
                captured_regions = self.screen_capture.capture_all_regions()
                if not captured_regions:
                    time.sleep(0.1)
                    continue
                
                # 2. Analyse des images
                game_info = self._analyze_game_state(captured_regions)
                if not game_info:
                    time.sleep(0.1)
                    continue
                
                # 3. Mise à jour de l'état du jeu
                self.game_state.update(game_info)
                
                # 4. Vérification si c'est notre tour
                if not self.game_state.is_my_turn:
                    time.sleep(0.1)
                    continue
                
                # 5. Prise de décision
                decision = self._make_decision()
                if decision:
                    # 6. Exécution de l'action
                    self._execute_action(decision)
                
                # Contrôle du FPS
                time.sleep(1.0 / self.config.getint('Display', 'capture_fps', fallback=10))
                
            except Exception as e:
                self.logger.error(f"Erreur boucle principale: {e}")
                time.sleep(1)
    
    def _monitor_loop(self):
        """Thread de surveillance et sécurité"""
        while self.running:
            try:
                # Vérifications de sécurité
                self._check_safety_conditions()
                
                # Mise à jour des statistiques
                self._update_stats()
                
                time.sleep(5)  # Vérification toutes les 5 secondes
                
            except Exception as e:
                self.logger.error(f"Erreur thread surveillance: {e}")
    
    def _analyze_game_state(self, captured_regions: Dict) -> Optional[Dict]:
        """Analyse l'état du jeu à partir des images capturées"""
        try:
            game_info = {}
            
            # Analyse des cartes
            if 'hand_area' in captured_regions:
                my_cards = self.image_analyzer.detect_cards(captured_regions['hand_area'])
                game_info['my_cards'] = my_cards
            
            if 'community_cards' in captured_regions:
                community_cards = self.image_analyzer.detect_cards(captured_regions['community_cards'])
                game_info['community_cards'] = community_cards
            
            # Analyse des boutons d'action
            if 'action_buttons' in captured_regions:
                buttons = self.button_detector.detect_available_actions(captured_regions['action_buttons'])
                game_info['available_actions'] = buttons
            
            # Analyse des stacks et mises
            game_info.update(self._analyze_stacks_and_bets(captured_regions))
            
            # Analyse de la position et des blinds
            game_info.update(self._analyze_position_and_blinds(captured_regions))
            
            return game_info
            
        except Exception as e:
            self.logger.error(f"Erreur analyse état jeu: {e}")
            return None
    
    def _analyze_stacks_and_bets(self, captured_regions: Dict) -> Dict:
        """Analyse les stacks et les mises"""
        info = {}
        
        try:
            # Stack du joueur
            if 'my_stack_area' in captured_regions:
                my_stack = self.image_analyzer.extract_text(captured_regions['my_stack_area'])
                info['my_stack'] = self._parse_stack_amount(my_stack)
            
            # Mise actuelle
            if 'current_bet_to_call' in captured_regions:
                bet_to_call = self.image_analyzer.extract_text(captured_regions['current_bet_to_call'])
                info['bet_to_call'] = self._parse_bet_amount(bet_to_call)
            
            # Pot
            if 'pot_area' in captured_regions:
                pot_text = self.image_analyzer.extract_text(captured_regions['pot_area'])
                info['pot_size'] = self._parse_stack_amount(pot_text)
            
        except Exception as e:
            self.logger.error(f"Erreur analyse stacks/mises: {e}")
        
        return info
    
    def _analyze_position_and_blinds(self, captured_regions: Dict) -> Dict:
        """Analyse la position et les blinds"""
        info = {}
        
        try:
            # Détection du bouton dealer
            if 'my_dealer_button' in captured_regions:
                info['my_is_dealer'] = self.image_analyzer.detect_dealer_button(captured_regions['my_dealer_button'])
            
            # Timer des blinds
            if 'blinds_timer' in captured_regions:
                timer_text = self.image_analyzer.extract_text(captured_regions['blinds_timer'])
                info['blinds_timer'] = self._parse_timer(timer_text)
            
        except Exception as e:
            self.logger.error(f"Erreur analyse position/blinds: {e}")
        
        return info
    
    def _make_decision(self) -> Optional[Dict]:
        """Prend une décision basée sur l'état du jeu"""
        try:
            # Utiliser la stratégie Spin & Rush par défaut
            # Basculer vers la stratégie générale seulement si ce n'est PAS un format Spin & Rush
            if not self._should_use_spin_rush_strategy():
                self.current_strategy = self.general_strategy
            else:
                self.current_strategy = self.spin_rush_strategy
            
            # Décision principale
            action = self.current_strategy.get_action_decision(self.game_state)
            
            # Calcul de la taille de mise
            bet_size = self.current_strategy.calculate_bet_size(action, self.game_state)
            
            return {
                'action': action,
                'bet_size': bet_size,
                'strategy': self.current_strategy.__class__.__name__
            }
            
        except Exception as e:
            self.logger.error(f"Erreur prise de décision: {e}")
            return None
    
    def _should_use_spin_rush_strategy(self) -> bool:
        """Détermine si on doit utiliser la stratégie Spin & Rush"""
        # Par défaut, utiliser Spin & Rush sauf si on détecte clairement un format différent
        # Logique pour détecter le format Spin & Rush
        # Par exemple: 3 joueurs, timer de blinds, stack 500
        
        # Si on a des informations claires sur le format
        if hasattr(self.game_state, 'num_players') and self.game_state.num_players > 3:
            return False  # Plus de 3 joueurs = poker standard
        
        if hasattr(self.game_state, 'my_stack') and self.game_state.my_stack > 1000:
            return False  # Stack élevé = probablement poker standard
        
        # Par défaut, utiliser Spin & Rush (plus agressif)
        return True
    
    def _execute_action(self, decision: Dict):
        """Exécute l'action décidée"""
        try:
            action = decision['action']
            bet_size = decision['bet_size']
            
            self.logger.info(f"Exécution: {action} (mise: {bet_size})")
            
            if action == Action.FOLD.value:
                self.automation.click_fold()
            elif action == Action.CALL.value:
                self.automation.click_call()
            elif action == Action.CHECK.value:
                self.automation.click_check()
            elif action == Action.RAISE.value or action == Action.BET.value:
                self.automation.click_raise(bet_size)
            elif action == Action.ALL_IN.value:
                self.automation.click_all_in()
            
            # Mise à jour des statistiques
            self.stats['hands_played'] += 1
            
        except Exception as e:
            self.logger.error(f"Erreur exécution action: {e}")
    
    def _parse_stack_amount(self, text: str) -> float:
        """Parse un montant de stack"""
        try:
            # Nettoyer le texte et extraire le nombre
            import re
            numbers = re.findall(r'[\d,]+\.?\d*', text.replace(' ', ''))
            if numbers:
                return float(numbers[0].replace(',', ''))
            return 0.0
        except:
            return 0.0
    
    def _parse_bet_amount(self, text: str) -> float:
        """Parse un montant de mise"""
        return self._parse_stack_amount(text)
    
    def _parse_timer(self, text: str) -> int:
        """Parse un timer"""
        try:
            import re
            numbers = re.findall(r'\d+', text)
            if numbers:
                return int(numbers[0])
            return 0
        except:
            return 0
    
    def _check_safety_conditions(self):
        """Vérifie les conditions de sécurité"""
        try:
            # Vérification du nombre de mains par heure
            session_duration = time.time() - self.stats['session_start']
            hands_per_hour = (self.stats['hands_played'] / session_duration) * 3600
            
            max_hands = self.config.getint('Safety', 'max_hands_per_hour', fallback=180)
            if hands_per_hour > max_hands:
                self.logger.warning(f"Trop de mains par heure: {hands_per_hour:.1f}")
                self.pause()
            
        except Exception as e:
            self.logger.error(f"Erreur vérification sécurité: {e}")
    
    def _update_stats(self):
        """Met à jour les statistiques"""
        try:
            # Calcul du profit
            if hasattr(self.game_state, 'my_stack') and hasattr(self.game_state, 'initial_stack'):
                current_profit = self.game_state.my_stack - self.game_state.initial_stack
                self.stats['total_profit'] = current_profit
            
        except Exception as e:
            self.logger.error(f"Erreur mise à jour stats: {e}")
    
    def _save_session_stats(self):
        """Sauvegarde les statistiques de session"""
        try:
            session_duration = time.time() - self.stats['session_start']
            
            stats_summary = {
                'session_duration_hours': session_duration / 3600,
                'hands_played': self.stats['hands_played'],
                'hands_per_hour': (self.stats['hands_played'] / session_duration) * 3600,
                'total_profit': self.stats['total_profit'],
                'win_rate': self.stats['hands_won'] / max(self.stats['hands_played'], 1)
            }
            
            self.logger.info(f"Statistiques de session: {stats_summary}")
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde stats: {e}")

def main():
    """Point d'entrée principal"""
    try:
        # Gestion des signaux pour arrêt propre
        def signal_handler(signum, frame):
            print("\nArrêt demandé...")
            if hasattr(main, 'agent'):
                main.agent.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Création et démarrage de l'agent
        agent = PokerAgent()
        main.agent = agent  # Référence pour signal_handler
        
        print("=== AGENT IA POKER ===")
        print("Appuyez sur Ctrl+C pour arrêter")
        print("Appuyez sur 'p' pour pause/reprise")
        
        agent.start()
        
    except Exception as e:
        print(f"Erreur critique: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 