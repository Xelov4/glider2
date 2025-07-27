"""
🤖 Agent IA Poker - Betclic Poker
================================

POINT D'ENTRÉE PRINCIPAL
========================

Ce fichier contient la classe PokerAgent qui orchestre toutes les fonctionnalités :
- Capture d'écran en temps réel
- Détection des cartes et boutons
- Analyse stratégique du jeu
- Prise de décision intelligente
- Exécution automatique des actions

ARCHITECTURE
============

1. INITIALISATION
   - Chargement de la configuration
   - Initialisation des modules (OCR, capture, etc.)
   - Calibration des régions d'écran

2. BOUCLE PRINCIPALE
   - Capture ultra-rapide (10ms)
   - Détection des éléments de jeu
   - Analyse stratégique
   - Prise de décision
   - Exécution d'action

3. STRATÉGIES
   - Spin & Rush (ultra-agressive)
   - Générale (équilibrée)
   - Adaptative (selon le contexte)

4. GESTION D'ÉTAT
   - Suivi des mains
   - Statistiques de session
   - Logging détaillé

FONCTIONNALITÉS CLÉS
====================

✅ Détection continue pendant le chargement
✅ Réactivité ultra-rapide (< 100ms)
✅ Stratégie Spin & Rush intégrée
✅ Validation robuste des données
✅ Logging détaillé pour debugging
✅ Gestion d'erreurs complète

VERSION: 2.0.0
DERNIÈRE MISE À JOUR: 2025-07-27
AUTEUR: Assistant IA
"""

import sys
import time
import logging
import signal
import threading
from typing import Dict, Optional, List
from pathlib import Path
import concurrent.futures
import json

# Import des modules unifiés
from modules.screen_capture import ScreenCapture
from modules.image_analysis import ImageAnalyzer
from modules.button_detector import ButtonDetector
from modules.game_state import GameState
from modules.strategy_engine import GeneralStrategy
from modules.aggressive_strategy import AggressiveStrategy
from modules.automation import AutomationEngine
from modules.advanced_ai_engine import AdvancedAIEngine
from modules.constants import Position, Action, GamePhase, DEFAULT_CONFIG

class PokerAgent:
    """
    Agent IA Poker principal - Version unifiée
    """
    
    def __init__(self, config_file: str = "config.ini"):
        """Initialise l'agent avec configuration optimisée"""
        self.config_file = config_file
        self.logger = self._setup_logging()
        self.config = self._load_config()
        
        # État de jeu avec surveillance continue
        self.current_hand_id = None
        self.previous_cards = []
        self.hand_start_time = None
        self.last_action_time = time.time()
        
        # NOUVEAU: Cache intelligent pour performance
        self.image_cache = {}
        self.decision_cache = {}
        self.last_capture_time = 0
        self.cache_ttl = 0.1  # 100ms TTL pour le cache
        
        # NOUVEAU: Métriques de performance
        self.performance_metrics = {
            'capture_times': [],
            'decision_times': [],
            'ocr_times': [],
            'total_cycles': 0,
            'avg_cycle_time': 0
        }
        
        # NOUVEAU: Threading pour parallélisation
        self.capture_thread = None
        self.analysis_thread = None
        self.captured_regions_queue = []
        self.analysis_results_queue = []
        
        # Stats en temps réel
        self.session_stats = {
            'hands_played': 0,
            'hands_won': 0,
            'total_profit': 0.0,
            'session_start': time.time(),
            'actions_taken': 0,
            'decisions_made': 0,
            'errors_count': 0
        }
        
        # NOUVEAU: Configuration de performance
        self.performance_config = {
            'capture_interval': 0.01,  # 10ms entre captures
            'decision_timeout': 0.05,  # 50ms max pour décision
            'cache_enabled': True,
            'parallel_processing': True,
            'ultra_fast_mode': True
        }
        
        # Initialisation des modules avec optimisation
        self.screen_capture = ScreenCapture()
        self.image_analyzer = ImageAnalyzer()
        self.button_detector = ButtonDetector()
        self.game_state = GameState()
        self.automation = AutomationEngine()
        self.advanced_ai = AdvancedAIEngine()
        
        # Stratégies avec cache
        self.aggressive_strategy = AggressiveStrategy()
        self.general_strategy = GeneralStrategy()
        self.current_strategy = None  # Stratégie actuelle pour décision instantanée
        
        # NOUVEAU: État de jeu optimisé
        self.game_state_cache = {
            'last_update': 0,
            'cached_state': None,
            'buttons_cache': [],
            'cards_cache': [],
            'stack_cache': 0,
            'pot_cache': 0
        }
        
        # NOUVEAU: Monitoring en temps réel
        self.monitoring_active = False
        self.last_performance_log = time.time()
        
        self.logger.info("🤖 Agent IA Poker initialisé avec optimisations de performance")
        
        # Configuration
        self.running = False
        self.paused = False
        
        # Threads pour performance
        self.decision_thread = None
        self.monitor_thread = None  # Pour compatibilité
        
    def _setup_logging(self) -> logging.Logger:
        """Configure le logging intelligent et structuré"""
        logger = logging.getLogger('PokerAgent')
        logger.setLevel(logging.INFO)
        
        # NOUVEAU: Formatter structuré
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Handler console avec filtrage intelligent
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # NOUVEAU: Filtre pour réduire le bruit
        class SmartFilter(logging.Filter):
            def filter(self, record):
                # Filtrer les messages trop verbeux
                noisy_patterns = [
                    'Aucune carte detectee',
                    'Stack non detecte',
                    'Timer non detecte',
                    'Cache hit',
                    'DEBUG'
                ]
                
                for pattern in noisy_patterns:
                    if pattern.lower() in record.getMessage().lower():
                        return False
                
                return True
        
        console_handler.addFilter(SmartFilter())
        logger.addHandler(console_handler)
        
        # NOUVEAU: Handler fichier pour debug complet
        file_handler = logging.FileHandler('poker_agent_debug.log', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # NOUVEAU: Handler performance
        perf_handler = logging.FileHandler('poker_agent_performance.log', mode='w')
        perf_handler.setLevel(logging.INFO)
        perf_formatter = logging.Formatter('%(asctime)s - PERFORMANCE - %(message)s')
        perf_handler.setFormatter(perf_formatter)
        logger.addHandler(perf_handler)
        
        return logger

    def log_performance_metrics(self):
        """Log les métriques de performance"""
        try:
            if not self.performance_metrics['capture_times']:
                return
            
            avg_capture = sum(self.performance_metrics['capture_times']) / len(self.performance_metrics['capture_times'])
            avg_decision = sum(self.performance_metrics['decision_times']) / len(self.performance_metrics['decision_times'])
            
            self.logger.info(f"PERF - Capture: {avg_capture:.3f}s, Décision: {avg_decision:.3f}s, "
                           f"Cycles: {self.performance_metrics['total_cycles']}")
            
        except Exception as e:
            self.logger.error(f"Erreur log performance: {e}")

    def log_game_state(self, game_state: Dict, action: str = None):
        """Log l'état du jeu de manière structurée"""
        try:
            if not game_state:
                return
            
            my_cards = game_state.get('my_cards', [])
            community_cards = game_state.get('community_cards', [])
            my_stack = game_state.get('my_stack', 0)
            pot_size = game_state.get('pot_size', 0)
            position = game_state.get('position', '?')
            timer = game_state.get('timer', 0)
            
            state_summary = f"GAME - Cartes: {my_cards}, Communes: {community_cards}, "
            state_summary += f"Stack: {my_stack}, Pot: {pot_size}, Pos: {position}, Timer: {timer}"
            
            if action:
                state_summary += f" -> {action}"
            
            self.logger.info(state_summary)
            
        except Exception as e:
            self.logger.error(f"Erreur log état jeu: {e}")
    
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
            
            # LANCEMENT IMMÉDIAT D'UNE PARTIE AU DÉMARRAGE
            self.logger.info("Lancement immediat d'une nouvelle partie...")
            self._launch_initial_game()
            
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
        """Boucle principale ultra-optimisée avec monitoring"""
        self.logger.info("🚀 Démarrage de la boucle principale ultra-optimisée")
        
        cycle_count = 0
        last_performance_log = time.time()
        
        try:
            while self.running:
                cycle_start = time.time()
                cycle_count += 1
                
                # NOUVEAU: Monitoring de performance
                if time.time() - last_performance_log > 10:  # Log toutes les 10s
                    self.log_performance_metrics()
                    last_performance_log = time.time()
                
                try:
                    # 1. CAPTURE ULTRA-RAPIDE
                    captured_regions = self._capture_ultra_fast()
                    if not captured_regions:
                        time.sleep(0.001)  # 1ms si pas de capture
                        continue
                    
                    # 2. DÉTECTION D'ÉTAT DE JEU
                    game_state = self._detect_game_state_intelligent(captured_regions)
                    
                    # 3. GESTION DES ÉTATS
                    if game_state == 'no_game':
                        self._handle_no_game(captured_regions)
                    elif game_state == 'hand_ended':
                        self._handle_hand_ended(captured_regions)
                    elif game_state == 'our_turn':
                        self._handle_our_turn_optimized(captured_regions)
                    elif game_state == 'game_active':
                        self._handle_game_active_optimized(captured_regions)
                    
                    # NOUVEAU: Métriques de cycle
                    cycle_time = time.time() - cycle_start
                    self.performance_metrics['total_cycles'] = cycle_count
                    
                    # NOUVEAU: Contrôle de performance
                    if cycle_time > 0.1:  # Plus de 100ms
                        self.logger.warning(f"Cycle lent: {cycle_time:.3f}s")
                    
                    # NOUVEAU: Intervalle adaptatif
                    if cycle_time < 0.01:  # Moins de 10ms
                        time.sleep(0.001)  # 1ms de pause
                    elif cycle_time < 0.05:  # Moins de 50ms
                        time.sleep(0.005)  # 5ms de pause
                    
                except Exception as e:
                    self.logger.error(f"Erreur cycle principal: {e}")
                    self.session_stats['errors_count'] += 1
                    time.sleep(0.01)  # 10ms en cas d'erreur
                
        except KeyboardInterrupt:
            self.logger.info("Arrêt demandé par l'utilisateur")
        except Exception as e:
            self.logger.error(f"Erreur boucle principale: {e}")
        finally:
            self.logger.info("Boucle principale terminée")

    def _handle_our_turn_optimized(self, captured_regions: Dict):
        """Gestion optimisée de notre tour"""
        try:
            # NOUVEAU: Décision ultra-rapide
            available_buttons = self._detect_individual_action_buttons(captured_regions)
            
            if available_buttons:
                # Analyser l'état du jeu rapidement
                game_state = self._analyze_complete_game_state(captured_regions)
                
                if game_state:
                    # Prendre une décision intelligente
                    decision = self._make_instant_decision(game_state, available_buttons)
                    
                    if decision:
                        # Log l'état et l'action
                        self.log_game_state(game_state, decision['action'])
                        
                        # Exécuter immédiatement
                        self._execute_action_immediately(decision)
                        self.session_stats['actions_taken'] += 1
                        
                        # NOUVEAU: Pause courte après action
                        time.sleep(0.05)  # 50ms
                
        except Exception as e:
            self.logger.error(f"Erreur gestion tour optimisée: {e}")

    def _handle_game_active_optimized(self, captured_regions: Dict):
        """Gestion optimisée du jeu actif"""
        try:
            # NOUVEAU: Analyse continue mais légère
            self._continuous_game_analysis_and_decision(captured_regions)
            
            # NOUVEAU: Détection de fin de main
            if self._detect_winner_crown(captured_regions):
                self.logger.info("Couronne de victoire détectée - Fin de main")
                self._handle_hand_ended(captured_regions)
                
        except Exception as e:
            self.logger.error(f"Erreur gestion jeu actif optimisée: {e}")
    
    def _monitor_loop(self):
        """Thread de surveillance et sécurité"""
        while self.running:
            try:
                # Vérifications de sécurité
                self._check_safety_conditions()
                
                # Mise à jour des statistiques
                self._update_stats()
                
                # Mise à jour des statistiques de session
                self._update_session_stats()
                
                # Monitoring des performances
                self._monitor_performance()
                
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
                
                # Debug: afficher les boutons détectés
                if buttons:
                    button_names = [btn.name for btn in buttons]
                    self.logger.info(f"🎮 Boutons détectés: {button_names}")
                else:
                    self.logger.debug("Aucun bouton d'action détecté")
            
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
            if 'my_current_bet' in captured_regions:
                my_bet = self.image_analyzer.extract_text(captured_regions['my_current_bet'])
                info['my_current_bet'] = self._parse_bet_amount(my_bet)
            
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
        """Prend une décision avec le moteur d'IA avancée"""
        try:
            # Préparer les données pour l'IA
            game_data = {
                'my_cards': self.game_state.my_cards,
                'community_cards': self.game_state.community_cards,
                'pot_size': self.game_state.pot_size,
                'call_amount': self.game_state.call_amount,
                'my_stack': self.game_state.my_stack,
                'position': self.game_state.my_position,
                'street': self.game_state.street,
                'available_actions': self.game_state.available_actions,
                'num_players': self.game_state.num_players,
                'big_blind': self.game_state.big_blind,
                'small_blind': self.game_state.small_blind
            }
            
            # Décision avec IA avancée
            decision = self.ai_decision.make_decision(game_data)
            
            self.logger.info(f"IA Decision: {decision['action']} - {decision['reasoning']}")
            self.logger.info(f"Temps decision: {decision['decision_time']:.3f}s")
            
            return {
                'action': decision['action'],
                'bet_size': decision['bet_size'],
                'reasoning': decision['reasoning'],
                'confidence': decision['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"Erreur décision IA: {e}")
            return None
    
    def _should_use_aggressive_strategy(self) -> bool:
        """Détermine si on doit utiliser la stratégie agressive"""
        # Toujours utiliser la stratégie agressive
        return True
    
    def _execute_action(self, decision: Dict):
        """Exécute l'action décidée par l'IA avec délai humain"""
        try:
            action = decision.get('action', 'fold')
            bet_size = decision.get('bet_size', 0)
            reasoning = decision.get('reasoning', '')
            confidence = decision.get('confidence', 0.0)
            
            self.logger.info(f"Execution: {action.upper()} - {reasoning}")
            self.logger.info(f"Confiance: {confidence:.2f}")
            
            # Délai humain basé sur la confiance
            if confidence > 0.8:
                delay = random.uniform(0.1, 0.3)  # Rapide si confiant
            elif confidence > 0.5:
                delay = random.uniform(0.3, 0.6)  # Moyen
            else:
                delay = random.uniform(0.6, 1.0)  # Lent si pas sûr
            
            time.sleep(delay)
            
            # Exécution de l'action
            if action == 'fold':
                self.automation.click_fold()
            elif action == 'check':
                self.automation.click_check()
            elif action == 'call':
                self.automation.click_call()
            elif action == 'raise':
                if bet_size > 0:
                    # Utiliser le bet slider pour la mise
                    self.automation.set_bet_amount(bet_size)
                self.automation.click_raise()
            elif action == 'all_in':
                self.automation.click_all_in()
            else:
                self.logger.warning(f"Action inconnue: {action} - FOLD par défaut")
                self.automation.click_fold()
            
            # Mise à jour des stats
            self._update_action_stats(action, bet_size)
            
        except Exception as e:
            self.logger.error(f"Erreur exécution action: {e}")
            # Fallback: FOLD
            self.automation.click_fold()
    
    def _update_action_stats(self, action: str, bet_size: float):
        """Met à jour les statistiques d'action"""
        try:
            # Mettre à jour les stats de l'IA
            if hasattr(self.ai_decision, 'update_opponent_stats'):
                # Pour l'instant, on met à jour nos propres stats
                self.ai_decision.update_opponent_stats('self', action, bet_size)
            
            # Mise à jour des stats de session
            if action in ['call', 'raise', 'all_in']:
                self.stats['hands_played'] += 1
            
        except Exception as e:
            self.logger.error(f"Erreur mise à jour stats: {e}")
    
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
    
    def _update_session_stats(self):
        """Mise à jour des statistiques de session"""
        try:
            session_duration = time.time() - self.session_stats['session_start']
            
            # Calculer les métriques
            hands_per_hour = (self.session_stats['hands_played'] / session_duration) * 3600 if session_duration > 0 else 0
            win_rate = (self.session_stats['hands_won'] / self.session_stats['hands_played']) * 100 if self.session_stats['hands_played'] > 0 else 0
            actions_per_hand = self.session_stats['actions_taken'] / self.session_stats['hands_played'] if self.session_stats['hands_played'] > 0 else 0
            
            # Performance moyenne
            avg_capture_time = sum(self.performance_metrics['capture_times']) / len(self.performance_metrics['capture_times']) if self.performance_metrics['capture_times'] else 0
            avg_decision_time = sum(self.performance_metrics['decision_times']) / len(self.performance_metrics['decision_times']) if self.performance_metrics['decision_times'] else 0
            
            # Log des statistiques
            self.logger.info(f"STATS - Session: {session_duration:.1f}s, Mains: {self.session_stats['hands_played']}, "
                           f"Victoires: {self.session_stats['hands_won']} ({win_rate:.1f}%), "
                           f"Actions: {self.session_stats['actions_taken']}, Erreurs: {self.session_stats['errors_count']}")
            
            self.logger.info(f"PERF - Capture: {avg_capture_time:.3f}s, Décision: {avg_decision_time:.3f}s, "
                           f"Cycles: {self.performance_metrics['total_cycles']}")
            
        except Exception as e:
            self.logger.error(f"Erreur mise à jour stats: {e}")

    def _save_session_stats(self):
        """Sauvegarde des statistiques de session"""
        try:
            session_duration = time.time() - self.session_stats['session_start']
            
            stats_data = {
                'session_duration': session_duration,
                'hands_played': self.session_stats['hands_played'],
                'hands_won': self.session_stats['hands_won'],
                'total_profit': self.session_stats['total_profit'],
                'actions_taken': self.session_stats['actions_taken'],
                'decisions_made': self.session_stats['decisions_made'],
                'errors_count': self.session_stats['errors_count'],
                'avg_capture_time': sum(self.performance_metrics['capture_times']) / len(self.performance_metrics['capture_times']) if self.performance_metrics['capture_times'] else 0,
                'avg_decision_time': sum(self.performance_metrics['decision_times']) / len(self.performance_metrics['decision_times']) if self.performance_metrics['decision_times'] else 0,
                'total_cycles': self.performance_metrics['total_cycles'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Sauvegarder dans un fichier JSON
            with open('session_stats.json', 'w') as f:
                json.dump(stats_data, f, indent=2)
            
            self.logger.info("Statistiques de session sauvegardées")
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde stats: {e}")

    def _monitor_performance(self):
        """Monitoring en temps réel des performances"""
        try:
            # Vérifier les performances critiques
            if self.performance_metrics['capture_times']:
                recent_captures = self.performance_metrics['capture_times'][-10:]  # 10 dernières captures
                avg_recent_capture = sum(recent_captures) / len(recent_captures)
                
                if avg_recent_capture > 0.1:  # Plus de 100ms
                    self.logger.warning(f"Performance capture dégradée: {avg_recent_capture:.3f}s")
                    self._recover_from_error('capture_error')
            
            if self.performance_metrics['decision_times']:
                recent_decisions = self.performance_metrics['decision_times'][-10:]  # 10 dernières décisions
                avg_recent_decision = sum(recent_decisions) / len(recent_decisions)
                
                if avg_recent_decision > 0.05:  # Plus de 50ms
                    self.logger.warning(f"Performance décision dégradée: {avg_recent_decision:.3f}s")
                    self._recover_from_error('decision_error')
            
            # Vérifier le taux d'erreurs
            if self.session_stats['errors_count'] > 10:
                self.logger.warning(f"Trop d'erreurs: {self.session_stats['errors_count']}")
                
        except Exception as e:
            self.logger.error(f"Erreur monitoring performance: {e}")

    def _launch_initial_game(self):
        """Lance immédiatement une nouvelle partie au démarrage de l'agent"""
        try:
            self.logger.info("Tentative de lancement immediat d'une partie...")
            
            # 1. Capturer l'écran pour détecter le bouton "New Hand"
            captured_regions = self.screen_capture.capture_all_regions()
            
            # 2. Essayer de cliquer sur "New Hand" s'il est présent
            if self._check_and_click_new_hand_button(captured_regions):
                self.logger.info("✅ Partie lancée avec succès - Attente 15 secondes...")
                time.sleep(15)  # Attendre que la partie se lance
                return True
            else:
                # 3. Si pas de bouton détecté, essayer une position par défaut
                self.logger.info("Aucun bouton 'New Hand' detecte - Tentative position par defaut...")
                
                # Position par défaut pour le bouton "New Hand"
                region_info = self.screen_capture.get_region_info('new_hand_button')
                if region_info:
                    x, y = region_info['x'], region_info['y']
                    width, height = region_info['width'], region_info['height']
                    
                    # Calculer le centre du bouton
                    center_x = x + width // 2
                    center_y = y + height // 2
                    
                    # Cliquer sur la position par défaut
                    self.logger.info(f"Clic sur position par defaut ({center_x}, {center_y})")
                    self.automation.click_at_position(center_x, center_y)
                    
                    self.logger.info("Clic effectue - DETECTION CONTINUE PENDANT LE CHARGEMENT...")
                    
                    # DÉTECTION CONTINUE PENDANT LE CHARGEMENT (15 secondes)
                    start_time = time.time()
                    while time.time() - start_time < 15:
                        try:
                            # Capture et analyse continue
                            captured_regions = self._capture_ultra_fast()
                            if captured_regions:
                                # Détecter les cartes et éléments de jeu
                                self._detect_and_log_cards(captured_regions)
                                
                                # Vérifier si la partie a commencé
                                if self._detect_game_active(captured_regions):
                                    self.logger.info("PARTIE DÉTECTÉE - PASSAGE EN MODE JEU!")
                                    return True
                            
                            time.sleep(0.1)  # 100ms entre les captures
                            
                        except Exception as e:
                            self.logger.debug(f"Erreur détection pendant chargement: {e}")
                            time.sleep(0.1)
                    
                    self.logger.info("Fin de l'attente - Passage en mode normal")
                    return True
                else:
                    self.logger.warning("⚠️ Impossible de lancer une partie - Aucune position trouvée")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Erreur lancement initial: {e}")
            return False

    def _check_and_click_new_hand_button(self, captured_regions: Dict) -> bool:
        """Vérifie si le bouton "New Hand" est présent et clique dessus si oui."""
        try:
            # 1. PRIORITÉ: Utiliser les coordonnées calibrées
            region_info = self.screen_capture.get_region_info('new_hand_button')
            if region_info:
                x, y = region_info['x'], region_info['y']
                width, height = region_info['width'], region_info['height']
                
                # Capturer la région calibrée
                new_hand_image = self.screen_capture.capture_region('new_hand_button')
                if new_hand_image is not None:
                    # VALIDATION OCR AVANT CLIC
                    text = self.image_analyzer.extract_text(new_hand_image)
                    self.logger.debug(f"Texte detecte dans new_hand_button: '{text}'")
                    
                    # Vérifier si le texte contient un mot-clé "New Hand"
                    new_hand_keywords = ['new hand', 'nouvelle main', 'rejouer', 'play again', 'start game']
                    found_keyword = None
                    
                    for keyword in new_hand_keywords:
                        if keyword.lower() in text.lower():
                            found_keyword = keyword
                            break
                    
                    if found_keyword:
                        self.logger.info(f"Bouton 'New Hand' confirme (mot-cle: '{found_keyword}')")
                        
                        # Calculer le centre du bouton calibré
                        center_x = x + width // 2
                        center_y = y + height // 2
                        
                        # Cliquer sur le bouton calibré
                        self.logger.info(f"CLIC sur 'New Hand' a ({center_x}, {center_y})")
                        self.automation.click_at_position(center_x, center_y)
                        
                        # 🚨 PASSAGE IMMÉDIAT EN MODE POKER ULTRA-RÉACTIF
                        self.logger.info("NOUVELLE PARTIE LANCEE - MODE POKER ULTRA-REACTIF ACTIVÉ!")
                        
                        # Reset des états pour la nouvelle partie
                        self.current_hand_id = None
                        self.previous_cards = []
                        self.hand_start_time = time.time()
                        self.decision_cache.clear()
                        
                        # Attendre juste 2 secondes pour que la partie se lance
                        time.sleep(2)
                        
                        return True
                    else:
                        self.logger.debug(f"Aucun mot-cle 'New Hand' trouve dans le texte: '{text}'")
                        return False
            
            # 2. FALLBACK: Si pas de région calibrée, ne pas chercher sur tout l'écran
            self.logger.debug("Region 'new_hand_button' non trouvee - pas de fallback OCR")
            return False
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la tentative de clic sur le bouton 'New Hand': {e}")
            return False

    def _check_and_click_resume_button(self, captured_regions: Dict) -> bool:
        """Vérifie si le bouton "Reprendre" est présent et clique dessus si oui."""
        try:
            # 1. PRIORITÉ: Utiliser les coordonnées calibrées
            region_info = self.screen_capture.get_region_info('resume_button')
            if region_info:
                x, y = region_info['x'], region_info['y']
                width, height = region_info['width'], region_info['height']
                
                # Capturer la région calibrée
                resume_image = self.screen_capture.capture_region('resume_button')
                if resume_image is not None:
                    # VALIDATION OCR AVANT CLIC
                    text = self.image_analyzer.extract_text(resume_image)
                    self.logger.debug(f"Texte detecte dans resume_button: '{text}'")
                    
                    # Vérifier si le texte contient un mot-clé "Reprendre"
                    resume_keywords = ['reprendre', 'resume', 'continue', 'rejoindre', 'repartir']
                    found_keyword = None
                    
                    for keyword in resume_keywords:
                        if keyword in text.lower():
                            found_keyword = keyword
                            break
                    
                    if found_keyword:
                        self.logger.info(f"Bouton 'Reprendre' confirme (mot-cle: '{found_keyword}')")
                        
                        # Calculer le centre du bouton calibré
                        center_x = x + width // 2
                        center_y = y + height // 2
                        
                        # Cliquer sur le bouton calibré
                        self.logger.info(f"CLIC sur 'Reprendre' a ({center_x}, {center_y})")
                        self.automation.click_at_position(center_x, center_y)
                        
                        # Attendre un peu pour que l'interface se stabilise
                        time.sleep(1)
                        
                        return True
                    else:
                        self.logger.debug(f"Aucun mot-cle 'Reprendre' trouve dans le texte: '{text}'")
                        return False
            
            # 2. FALLBACK: Si pas de région calibrée, ne pas chercher sur tout l'écran
            self.logger.debug("Region 'resume_button' non trouvee - pas de fallback OCR")
            return False
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la tentative de clic sur le bouton 'Reprendre': {e}")
            return False

    def _detect_game_state_intelligent(self, captured_regions: Dict) -> str:
        """Détection intelligente de l'état de jeu selon les règles d'or de Betclic"""
        try:
            # 1. VÉRIFICATION PRIORITAIRE: BOUTONS D'ACTION VISIBLES
            if 'action_buttons' in captured_regions:
                buttons = self.button_detector.detect_available_actions(captured_regions['action_buttons'])
                if buttons:
                    # 🎯 BOUTONS VISIBLES = C'EST NOTRE TOUR (10 secondes max)
                    button_names = [btn.name for btn in buttons]
                    self.logger.info(f"🎯 BOUTONS DÉTECTÉS: {button_names} - C'EST NOTRE TOUR!")
                    return "OUR_TURN"
            
            # 2. VÉRIFICATION: FIN DE MAIN
            if self._detect_hand_end(captured_regions):
                self.logger.info("🏁 FIN DE MAIN DÉTECTÉE")
                return "HAND_ENDED"
            
            # 3. VÉRIFICATION: ÉLÉMENTS DE JEU PRÉSENTS
            game_elements_present = self._check_game_elements_present(captured_regions)
            
            if game_elements_present:
                # 🎮 ÉLÉMENTS DE JEU PRÉSENTS = PARTIE EN COURS
                # MAIS on va analyser plus en détail dans _handle_game_active
                return "GAME_ACTIVE"
            else:
                # 🚀 PAS D'ÉLÉMENTS = PAS DE PARTIE
                return "NO_GAME"
                
        except Exception as e:
            self.logger.error(f"Erreur détection état: {e}")
            return "ERROR"
    
    def _detect_hand_end(self, captured_regions: Dict) -> bool:
        """Détecte si la main est terminée"""
        try:
            # 1. Vérifier les messages de fin de main avec OCR
            if 'pot_area' in captured_regions:
                pot_text = self.image_analyzer.extract_text(captured_regions['pot_area'])
                if pot_text:
                    # Chercher des mots-clés de fin de main
                    end_keywords = ['gagné', 'perdu', 'won', 'lost', 'victoire', 'défaite']
                    for keyword in end_keywords:
                        if keyword.lower() in pot_text.lower():
                            return True
            
            # 2. Vérifier les changements de stack (si on a les données précédentes)
            if hasattr(self, 'previous_stack') and 'my_stack_area' in captured_regions:
                current_stack = self._parse_stack_amount(
                    self.image_analyzer.extract_text(captured_regions['my_stack_area'])
                )
                if current_stack != self.previous_stack:
                    self.logger.info(f"💰 Changement de stack détecté: {self.previous_stack} → {current_stack}")
                    self.previous_stack = current_stack
                    return True
            
            # 3. Vérifier l'absence de cartes communautaires (main terminée)
            if 'community_cards' in captured_regions:
                community_cards = self.image_analyzer.detect_cards(captured_regions['community_cards'])
                if not community_cards and hasattr(self, 'had_community_cards') and self.had_community_cards:
                    self.logger.info("🃏 Cartes communautaires disparues - main terminée")
                    return True
                self.had_community_cards = bool(community_cards)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Erreur détection fin de main: {e}")
            return False
    
    def _check_game_elements_present(self, captured_regions: Dict) -> bool:
        """Vérifie si les éléments de jeu sont présents (partie en cours)"""
        try:
            elements_present = 0
            total_elements = 0
            
            # Vérifier les éléments essentiels
            essential_elements = [
                'hand_area',      # Nos cartes
                'community_cards', # Cartes communes
                'pot_area',       # Pot
                'my_stack_area',  # Notre stack
                'opponent1_stack_area', # Stack adversaire 1
                'opponent2_stack_area'  # Stack adversaire 2
            ]
            
            for element in essential_elements:
                total_elements += 1
                if element in captured_regions:
                    # Vérifier que l'image n'est pas vide
                    img = captured_regions[element]
                    if img is not None and img.size > 0:
                        elements_present += 1
            
            # Si au moins 4 éléments sur 6 sont présents, la partie est en cours
            return elements_present >= 4
            
        except Exception as e:
            self.logger.error(f"Erreur vérification éléments: {e}")
            return False
    
    def _handle_our_turn(self, captured_regions: Dict):
        """Gère notre tour - ACTION ULTRA-RAPIDE REQUISE (10 secondes max)"""
        try:
            self.logger.info("URGENT: NOTRE TOUR - ACTION IMMÉDIATE!")
            
            # ANALYSE CRITIQUE SEULEMENT
            critical_data = self._analyze_critical_game_data(captured_regions)
            
            if critical_data:
                # DÉCISION ULTRA-RAPIDE
                start_time = time.time()
                decision = self._make_ultra_fast_decision(critical_data)
                decision_time = time.time() - start_time
                
                self.logger.info(f"DECISION ULTRA-RAPIDE: {decision['action'].upper()} en {decision_time:.3f}s")
                
                # EXÉCUTION IMMÉDIATE
                self._execute_action_immediately(decision)
            else:
                self.logger.warning("DONNÉES CRITIQUES MANQUANTES - FOLD IMMÉDIAT")
                self.automation.click_fold()
                
        except Exception as e:
            self.logger.error(f"Erreur gestion notre tour: {e}")
            # Fallback: FOLD immédiat pour éviter timeout
            self.automation.click_fold()
    
    def _handle_game_active(self, captured_regions: Dict):
        """Gère la partie en cours - DÉTECTION ULTRA-RAPIDE DES BOUTONS D'ACTION"""
        try:
            # 🎯 PRIORITÉ ABSOLUE: BOUTONS D'ACTION (10 secondes max)
            if 'action_buttons' in captured_regions:
                buttons = self.button_detector.detect_available_actions(captured_regions['action_buttons'])
                if buttons:
                    # 🚨 BOUTONS DÉTECTÉS = C'EST NOTRE TOUR - ACTION IMMÉDIATE !
                    button_names = [btn.name for btn in buttons]
                    # Boutons détectés
                    
                    # ANALYSE RAPIDE SEULEMENT DES DONNÉES CRITIQUES
                    critical_data = self._analyze_critical_game_data(captured_regions)
                    
                    if critical_data:
                        # DÉCISION ULTRA-RAPIDE
                        start_time = time.time()
                        decision = self._make_ultra_fast_decision(critical_data)
                        decision_time = time.time() - start_time
                        
                        self.logger.info(f"DECISION ULTRA-RAPIDE: {decision['action'].upper()} en {decision_time:.3f}s")
                        
                        # EXÉCUTION IMMÉDIATE
                        self._execute_action_immediately(decision)
                    else:
                        self.logger.warning("DONNÉES CRITIQUES MANQUANTES - FOLD")
                        self.automation.click_fold()
                else:
                    # Pas notre tour - surveillance ultra-rapide
                    time.sleep(0.05)  # 50ms seulement
            else:
                # Pas de boutons - surveillance ultra-rapide
                time.sleep(0.05)  # 50ms seulement
                
        except Exception as e:
            self.logger.error(f"Erreur gestion ultra-réactive: {e}")
            # Fallback: FOLD immédiat
            self.automation.click_fold()
    
    def _analyze_critical_game_data(self, captured_regions: Dict) -> Optional[Dict]:
        """Analyse UNIQUEMENT les données critiques avec calculs avancés"""
        try:
            critical_data = {}
            
            # 1. NOS CARTES (CRITIQUE) + DÉTECTION CHANGEMENT
            if 'hand_area' in captured_regions:
                my_cards = self.image_analyzer.detect_cards(captured_regions['hand_area'])
                if my_cards:
                    critical_data['my_cards'] = my_cards
                    # DÉTECTION CHANGEMENT DE MAIN
                    self._detect_hand_change(my_cards)
            
            # 2. NOTRE STACK (CRITIQUE)
            if 'my_stack_area' in captured_regions:
                stack_text = self.image_analyzer.extract_text(captured_regions['my_stack_area'])
                if stack_text:
                    stack_amount = self._parse_stack_amount(stack_text)
                    critical_data['my_stack'] = stack_amount
            
            # 3. POT (CRITIQUE)
            if 'pot_area' in captured_regions:
                pot_text = self.image_analyzer.extract_text(captured_regions['pot_area'])
                if pot_text:
                    pot_amount = self._parse_bet_amount(pot_text)
                    critical_data['pot_size'] = pot_amount
            
            # 4. CARTES COMMUNAUTAIRES (si présentes)
            if 'community_cards' in captured_regions:
                community_cards = self.image_analyzer.detect_cards(captured_regions['community_cards'])
                if community_cards:
                    critical_data['community_cards'] = community_cards
            
            # 5. MISE À CALL (CRITIQUE)
            if 'my_current_bet' in captured_regions:
                bet_text = self.image_analyzer.extract_text(captured_regions['my_current_bet'])
                if bet_text:
                    call_amount = self._parse_bet_amount(bet_text)
                    critical_data['call_amount'] = call_amount
            
            # 6. CALCULS AVANCÉS SI DONNÉES SUFFISANTES
            if len(critical_data) >= 3:  # Au moins cartes + stack + pot
                critical_data.update(self._calculate_advanced_metrics(critical_data))
            
            return critical_data if len(critical_data) >= 2 else None
            
        except Exception as e:
            self.logger.error(f"Erreur analyse critique: {e}")
            return None
    
    def _calculate_advanced_metrics(self, data: Dict) -> Dict:
        """Calcule les métriques avancées en temps réel"""
        try:
            advanced_metrics = {}
            
            my_cards = data.get('my_cards', [])
            my_stack = data.get('my_stack', 0)
            pot_size = data.get('pot_size', 0)
            call_amount = data.get('call_amount', 0)
            community_cards = data.get('community_cards', [])
            
            # 1. EQUITY CALCUL
            if my_cards and community_cards:
                equity = self._calculate_equity_vs_range(my_cards, community_cards)
                advanced_metrics['equity'] = equity
            
            # 2. POT ODDS
            if pot_size > 0 and call_amount > 0:
                pot_odds = pot_size / call_amount
                advanced_metrics['pot_odds'] = pot_odds
                advanced_metrics['pot_odds_profitable'] = pot_odds > 1.0
            
            # 3. IMPLIED ODDS
            if my_stack > 0 and pot_size > 0:
                implied_odds = self._calculate_implied_odds(my_stack, pot_size, data.get('equity', 0.5))
                advanced_metrics['implied_odds'] = implied_odds
            
            # 4. STACK-TO-POT RATIO
            if pot_size > 0:
                spr = my_stack / pot_size
                advanced_metrics['spr'] = spr
                advanced_metrics['spr_category'] = self._categorize_spr(spr)
            
            # 5. BET SIZING OPTIMAL
            if pot_size > 0:
                optimal_bet = self._calculate_optimal_bet_size(pot_size, my_stack, data.get('equity', 0.5))
                advanced_metrics['optimal_bet'] = optimal_bet
            
            return advanced_metrics
            
        except Exception as e:
            self.logger.error(f"Erreur calcul métriques avancées: {e}")
            return {}
    
    def _calculate_equity_vs_range(self, my_cards: List[str], community_cards: List[str]) -> float:
        """Calcule l'équité vs une range d'adversaire (simplifié)"""
        try:
            if not my_cards:
                return 0.0
            
            # Évaluation rapide de la force
            hand_strength = self._quick_hand_evaluation(my_cards, community_cards)
            
            # Ajustement selon le nombre de cartes communautaires
            if len(community_cards) == 0:  # Preflop
                return hand_strength * 0.8  # Équité préflop
            elif len(community_cards) == 3:  # Flop
                return hand_strength * 0.9  # Équité flop
            elif len(community_cards) == 4:  # Turn
                return hand_strength * 0.95  # Équité turn
            else:  # River
                return hand_strength  # Équité river
            
        except Exception as e:
            self.logger.error(f"Erreur calcul équité: {e}")
            return 0.5
    
    def _calculate_implied_odds(self, stack: float, pot: float, equity: float) -> float:
        """Calcule les implied odds"""
        try:
            if equity <= 0:
                return 0.0
            
            # Formule simplifiée: (stack - pot) * equity / pot
            if pot > 0:
                return (stack - pot) * equity / pot
            else:
                return 0.0
            
        except Exception as e:
            self.logger.error(f"Erreur calcul implied odds: {e}")
            return 0.0
    
    def _categorize_spr(self, spr: float) -> str:
        """Catégorise le SPR (Stack-to-Pot Ratio)"""
        if spr < 1:
            return 'short'
        elif spr < 3:
            return 'medium'
        elif spr < 6:
            return 'deep'
        else:
            return 'very_deep'
    
    def _calculate_optimal_bet_size(self, pot: float, stack: float, equity: float) -> float:
        """Calcule la taille de mise optimale"""
        try:
            if equity > 0.8:  # Main très forte
                sizing = 0.75  # 75% du pot
            elif equity > 0.6:  # Main forte
                sizing = 0.6  # 60% du pot
            elif equity > 0.4:  # Main moyenne
                sizing = 0.4  # 40% du pot
            else:  # Main faible
                sizing = 0.25  # 25% du pot (bluff)
            
            optimal_bet = pot * sizing
            
            # Limiter à notre stack
            if optimal_bet > stack:
                optimal_bet = stack
            
            return optimal_bet
            
        except Exception as e:
            self.logger.error(f"Erreur calcul bet sizing: {e}")
            return pot * 0.5  # 50% par défaut
    
    def _make_ultra_fast_decision(self, critical_data: Dict) -> Dict:
        """Prend une décision ultra-rapide avec logique avancée"""
        try:
            my_cards = critical_data.get('my_cards', [])
            my_stack = critical_data.get('my_stack', 0)
            pot_size = critical_data.get('pot_size', 0)
            call_amount = critical_data.get('call_amount', 0)
            community_cards = critical_data.get('community_cards', [])
            
            # LOGIQUE ULTRA-RAPIDE AVANCÉE
            if not my_cards or my_stack <= 0:
                return {'action': 'fold', 'reasoning': 'Données manquantes'}
            
            # 1. CALCULS CRITIQUES
            hand_strength = self._calculate_hand_strength_ultra_fast(my_cards, community_cards)
            position = self._get_current_position()
            spr = self._calculate_stack_to_pot_ratio(my_stack, pot_size)
            pot_odds = self._calculate_pot_odds(pot_size, call_amount)
            
            # 2. DÉCISION BASÉE SUR LA POSITION ET LA FORCE
            if position == 'BTN':  # Dealer - plus agressif
                return self._btn_decision(hand_strength, spr, pot_odds, call_amount)
            elif position == 'BB':  # Big Blind - défensif
                return self._bb_decision(hand_strength, spr, pot_odds, call_amount)
            else:  # UTG - conservateur
                return self._utg_decision(hand_strength, spr, pot_odds, call_amount)
            
        except Exception as e:
            self.logger.error(f"Erreur décision ultra-rapide: {e}")
            return {'action': 'fold', 'reasoning': f'Erreur: {e}'}
    
    def _get_current_position(self) -> str:
        """Détermine la position actuelle"""
        try:
            # Logique simplifiée - à améliorer avec détection des boutons dealer
            # Pour l'instant, on utilise une heuristique basée sur le temps
            if hasattr(self, 'hand_start_time') and self.hand_start_time:
                elapsed = time.time() - self.hand_start_time
                if elapsed < 2:  # Début de main
                    return 'UTG'
                elif elapsed < 5:  # Milieu
                    return 'BB'
                else:  # Fin
                    return 'BTN'
            return 'UTG'  # Par défaut
            
        except Exception as e:
            self.logger.error(f"Erreur position: {e}")
            return 'UTG'
    
    def _calculate_stack_to_pot_ratio(self, stack: float, pot: float) -> float:
        """Calcule le ratio Stack-to-Pot"""
        if pot <= 0:
            return float('inf')
        return stack / pot
    
    def _calculate_pot_odds(self, pot: float, call_amount: float) -> float:
        """Calcule les pot odds"""
        if call_amount <= 0:
            return float('inf')
        return pot / call_amount
    
    def _btn_decision(self, hand_strength: float, spr: float, pot_odds: float, call_amount: float) -> Dict:
        """Décision pour le bouton (dealer) - agressif"""
        if hand_strength > 0.8:  # Main très forte
            return {'action': 'raise', 'reasoning': f'BTN main très forte ({hand_strength:.2f})'}
        elif hand_strength > 0.6:  # Main forte
            if spr > 3:  # Deep stack
                return {'action': 'raise', 'reasoning': f'BTN main forte deep ({hand_strength:.2f})'}
            else:
                return {'action': 'all_in', 'reasoning': f'BTN main forte short ({hand_strength:.2f})'}
        elif hand_strength > 0.4:  # Main moyenne
            if call_amount == 0:  # Check possible
                return {'action': 'check', 'reasoning': f'BTN main moyenne ({hand_strength:.2f})'}
            elif pot_odds > 2:  # Bonnes pot odds
                return {'action': 'call', 'reasoning': f'BTN main moyenne bonnes odds ({hand_strength:.2f})'}
            else:
                return {'action': 'fold', 'reasoning': f'BTN main moyenne mauvaises odds ({hand_strength:.2f})'}
        else:  # Main faible
            if call_amount == 0:  # Check possible
                return {'action': 'check', 'reasoning': f'BTN main faible ({hand_strength:.2f})'}
            else:
                return {'action': 'fold', 'reasoning': f'BTN main faible ({hand_strength:.2f})'}
    
    def _bb_decision(self, hand_strength: float, spr: float, pot_odds: float, call_amount: float) -> Dict:
        """Décision pour la big blind - défensif"""
        if hand_strength > 0.9:  # Main premium
            return {'action': 'raise', 'reasoning': f'BB main premium ({hand_strength:.2f})'}
        elif hand_strength > 0.7:  # Main forte
            if spr > 2:  # Deep stack
                return {'action': 'raise', 'reasoning': f'BB main forte deep ({hand_strength:.2f})'}
            else:
                return {'action': 'all_in', 'reasoning': f'BB main forte short ({hand_strength:.2f})'}
        elif hand_strength > 0.5:  # Main moyenne
            if pot_odds > 3:  # Très bonnes pot odds
                return {'action': 'call', 'reasoning': f'BB main moyenne bonnes odds ({hand_strength:.2f})'}
            else:
                return {'action': 'fold', 'reasoning': f'BB main moyenne mauvaises odds ({hand_strength:.2f})'}
        else:  # Main faible
            if call_amount == 0:  # Check possible
                return {'action': 'check', 'reasoning': f'BB main faible ({hand_strength:.2f})'}
            else:
                return {'action': 'fold', 'reasoning': f'BB main faible ({hand_strength:.2f})'}
    
    def _utg_decision(self, hand_strength: float, spr: float, pot_odds: float, call_amount: float) -> Dict:
        """Décision pour UTG - conservateur"""
        if hand_strength > 0.9:  # Main premium
            return {'action': 'raise', 'reasoning': f'UTG main premium ({hand_strength:.2f})'}
        elif hand_strength > 0.8:  # Main très forte
            if spr > 4:  # Très deep stack
                return {'action': 'raise', 'reasoning': f'UTG main très forte deep ({hand_strength:.2f})'}
            else:
                return {'action': 'all_in', 'reasoning': f'UTG main très forte short ({hand_strength:.2f})'}
        elif hand_strength > 0.6:  # Main forte
            if pot_odds > 4:  # Excellentes pot odds
                return {'action': 'call', 'reasoning': f'UTG main forte excellentes odds ({hand_strength:.2f})'}
            else:
                return {'action': 'fold', 'reasoning': f'UTG main forte mauvaises odds ({hand_strength:.2f})'}
        else:  # Main faible/moyenne
            if call_amount == 0:  # Check possible
                return {'action': 'check', 'reasoning': f'UTG main faible ({hand_strength:.2f})'}
            else:
                return {'action': 'fold', 'reasoning': f'UTG main faible ({hand_strength:.2f})'}
    
    def _quick_hand_evaluation(self, my_cards: List[str], community_cards: List[str]) -> float:
        """Évaluation ultra-rapide de la force de main (0.0 à 1.0)"""
        try:
            if not my_cards:
                return 0.0
            
            # ÉVALUATION PRÉFLOP
            if not community_cards:
                return self._preflop_hand_strength(my_cards)
            
            # ÉVALUATION POSTFLOP (simplifiée)
            all_cards = my_cards + community_cards
            return self._postflop_hand_strength(all_cards)
            
        except Exception as e:
            self.logger.error(f"Erreur évaluation main: {e}")
            return 0.0
    
    def _preflop_hand_strength(self, my_cards: List[str]) -> float:
        """Force de main préflop (0.0 à 1.0)"""
        if len(my_cards) != 2:
            return 0.0
        
        card1, card2 = my_cards[0], my_cards[1]
        
        # PAIRES
        if card1[0] == card2[0]:
            if card1[0] in ['A', 'K', 'Q']:
                return 0.9  # Paire premium
            elif card1[0] in ['J', 'T', '9']:
                return 0.7  # Paire forte
            else:
                return 0.5  # Paire moyenne
        
        # MAINS SUIVIES
        if card1[0] == card2[0]:  # Même rang
            if card1[0] in ['A', 'K', 'Q']:
                return 0.8  # Suivie premium
            else:
                return 0.6  # Suivie normale
        
        # MAINS COULEUR
        if len(card1) > 1 and len(card2) > 1 and card1[1] == card2[1]:
            if card1[0] in ['A', 'K', 'Q'] and card2[0] in ['A', 'K', 'Q']:
                return 0.7  # Couleur premium
            else:
                return 0.5  # Couleur normale
        
        # MAINS DÉCONNECTÉES
        if card1[0] in ['A', 'K'] and card2[0] in ['A', 'K', 'Q']:
            return 0.6  # Main forte
        else:
            return 0.3  # Main faible
    
    def _postflop_hand_strength(self, all_cards: List[str]) -> float:
        """Force de main postflop (simplifiée)"""
        # Détection de patterns simples
        ranks = [card[0] for card in all_cards]
        
        # PAIRES
        for rank in set(ranks):
            if ranks.count(rank) >= 2:
                return 0.6  # Au moins une paire
        
        # HAUTE CARTE
        if 'A' in ranks or 'K' in ranks:
            return 0.4  # Haute carte
        
        return 0.2  # Main faible
    
    def _execute_action_immediately(self, decision: Dict):
        """Exécute l'action immédiatement sans délai"""
        try:
            action = decision.get('action', 'fold')
            reasoning = decision.get('reasoning', '')
            
            self.logger.info(f"Action: {action.upper()} - {reasoning}")
            
            # Mapper l'action vers le bouton correspondant
            action_to_button = {
                'fold': 'fold_button',
                'call': 'call_button', 
                'raise': 'raise_button',
                'check': 'check_button',
                'all_in': 'all_in_button'
            }
            
            if action in action_to_button:
                button_region = action_to_button[action]
                # Cliquer directement sur la région du bouton
                self._click_button_region(button_region)
            else:
                self.logger.warning(f"Action inconnue: {action}")
                self.automation.click_fold()
                
        except Exception as e:
            self.logger.error(f"Erreur exécution immédiate: {e}")
            self.automation.click_fold()
    
    def _click_button_region(self, button_region: str):
        """Clique sur une région de bouton spécifique"""
        try:
            # Obtenir les coordonnées de la région
            region_data = self.screen_capture.get_region_coordinates(button_region)
            if region_data:
                x, y, width, height = region_data
                # Cliquer au centre du bouton
                center_x = x + width // 2
                center_y = y + height // 2
                self.automation.click_at_position(center_x, center_y)
                self.logger.info(f"Clic sur {button_region} à ({center_x}, {center_y})")
            else:
                self.logger.error(f"Région {button_region} non trouvée")
        except Exception as e:
            self.logger.error(f"Erreur clic sur {button_region}: {e}")
    
    def _handle_hand_ended(self, captured_regions: Dict):
        """Gère la fin de main - attente du bouton 'New Hand'"""
        try:
            # Main terminée - chercher le bouton "New Hand"
            if self._check_and_click_new_hand_button(captured_regions):
                self.logger.info("✅ Nouvelle partie lancée après fin de main!")
            else:
                self.logger.debug("⏳ Attente du bouton 'New Hand'...")
                
        except Exception as e:
            self.logger.error(f"Erreur gestion fin de main: {e}")
    
    def _handle_no_game(self, captured_regions: Dict):
        """Gère l'absence de partie - lancement proactif"""
        try:
            # Pas de partie - essayer de lancer une nouvelle
            self.logger.info("🚀 Pas de partie - tentative de lancement...")
            if self._check_and_click_new_hand_button(captured_regions):
                self.logger.info("✅ Nouvelle partie lancée!")
                    
        except Exception as e:
            self.logger.error(f"Erreur gestion pas de partie: {e}")
    
    def _validate_game_data_quick(self, game_info: Dict) -> bool:
        """Validation rapide des données de jeu - VERSION PERMISSIVE"""
        try:
            # Vérifications essentielles
            if not game_info:
                return False
            
            # Vérifier qu'on a au moins un stack (même par défaut)
            if game_info.get('my_stack', 0) <= 0:
                self.logger.debug("Stack invalide")
                return False
            
            # Vérifier qu'on a des cartes (même partielles)
            my_cards = game_info.get('my_cards', [])
            if not isinstance(my_cards, list):
                self.logger.debug("my_cards doit être une liste")
                return False
            
            # Validation réussie - plus permissive
            self.logger.debug("Validation des données de jeu réussie")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur validation rapide: {e}")
            return False
    
    def _execute_action_with_minimal_delay(self, decision: Dict):
        """Exécute l'action avec délai minimal (pour paraître humain mais rapide)"""
        try:
            import random
            
            # Délai minimal mais humain (0.2-0.8 secondes)
            delay = random.uniform(0.2, 0.8)
            time.sleep(delay)
            
            # Exécution de l'action
            self._execute_action(decision)
            
        except Exception as e:
            self.logger.error(f"Erreur exécution action: {e}")

    def _capture_ultra_fast(self) -> Optional[Dict]:
        """
        Capture ultra-rapide avec cache intelligent et parallélisation
        """
        start_time = time.time()
        
        try:
            # NOUVEAU: Vérifier le cache d'abord
            if self.performance_config['cache_enabled']:
                cache_age = time.time() - self.last_capture_time
                if cache_age < self.cache_ttl and self.image_cache:
                    self.logger.debug(f"Cache hit: {cache_age:.3f}s")
                    return self.image_cache.copy()
            
            # NOUVEAU: Capture optimisée avec régions prioritaires
            critical_regions = [
                'fold_button', 'call_button', 'raise_button', 'check_button', 'all_in_button',
                'hand_area', 'community_cards', 'pot_area', 'my_stack_area'
            ]
            
            captured_regions = {}
            
            # Capture parallèle des régions critiques
            if self.performance_config['parallel_processing']:
                captured_regions = self._capture_regions_parallel(critical_regions)
            else:
                captured_regions = self._capture_regions_sequential(critical_regions)
            
            # NOUVEAU: Validation et nettoyage
            valid_regions = {}
            for region_name, image in captured_regions.items():
                if image is not None and image.size > 0:
                    valid_regions[region_name] = image
            
            # Mettre à jour le cache
            if valid_regions:
                self.image_cache = valid_regions.copy()
                self.last_capture_time = time.time()
                
                # NOUVEAU: Métriques de performance
                capture_time = time.time() - start_time
                self.performance_metrics['capture_times'].append(capture_time)
                if len(self.performance_metrics['capture_times']) > 100:
                    self.performance_metrics['capture_times'].pop(0)
                
                self.logger.debug(f"Capture ultra-rapide: {len(valid_regions)} régions en {capture_time:.3f}s")
                return valid_regions
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erreur capture ultra-rapide: {e}")
            self.session_stats['errors_count'] += 1
            return None

    def _capture_regions_parallel(self, regions: List[str]) -> Dict:
        """
        Capture parallèle des régions pour performance maximale
        """
        import concurrent.futures
        
        captured_regions = {}
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Lancer les captures en parallèle
                future_to_region = {
                    executor.submit(self.screen_capture.capture_region, region): region 
                    for region in regions
                }
                
                # Collecter les résultats
                for future in concurrent.futures.as_completed(future_to_region, timeout=0.05):
                    region = future_to_region[future]
                    try:
                        image = future.result()
                        if image is not None:
                            captured_regions[region] = image
                    except Exception as e:
                        self.logger.debug(f"Erreur capture {region}: {e}")
            
            return captured_regions
            
        except Exception as e:
            self.logger.error(f"Erreur capture parallèle: {e}")
            return {}

    def _capture_regions_sequential(self, regions: List[str]) -> Dict:
        """
        Capture séquentielle des régions (fallback)
        """
        captured_regions = {}
        
        for region in regions:
            try:
                image = self.screen_capture.capture_region(region)
                if image is not None:
                    captured_regions[region] = image
            except Exception as e:
                self.logger.debug(f"Erreur capture {region}: {e}")
        
        return captured_regions
    
    def _detect_hand_change(self, current_cards: List[str]) -> bool:
        """Détecte si on a changé de main"""
        try:
            if not current_cards:
                return False
            
            # Comparer avec les cartes précédentes
            if current_cards != self.previous_cards:
                if self.previous_cards:  # Pas la première main
                    self.logger.info(f"NOUVELLE MAIN DÉTECTÉE: {self.previous_cards} → {current_cards}")
                    self._on_new_hand()
                
                self.previous_cards = current_cards.copy()
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Erreur détection changement main: {e}")
            return False
    
    def _on_new_hand(self):
        """Appelé quand une nouvelle main est détectée"""
        try:
            self.current_hand_id = f"hand_{int(time.time())}"
            self.hand_start_time = time.time()
            
            # Reset des caches pour la nouvelle main
            self.decision_cache.clear()
            
            # Mise à jour des stats
            self.session_stats['hands_played'] += 1
            
            self.logger.info(f"NOUVELLE MAIN DETECTEE (ID: {self.current_hand_id})")
            
        except Exception as e:
            self.logger.error(f"Erreur nouvelle main: {e}")
    
    def _calculate_hand_strength_ultra_fast(self, my_cards: List[str], community_cards: List[str]) -> float:
        """Calcul ultra-rapide de la force de main avec cache"""
        try:
            # Cache key
            cache_key = f"{''.join(my_cards)}_{''.join(community_cards)}"
            
            if cache_key in self.decision_cache:
                return self.decision_cache[cache_key]
            
            # Calcul rapide
            strength = self._quick_hand_evaluation(my_cards, community_cards)
            
            # Cache le résultat
            self.decision_cache[cache_key] = strength
            
            return strength
            
        except Exception as e:
            self.logger.error(f"Erreur calcul force main: {e}")
            return 0.0

    def _analyze_game_elements_ultra_fast(self, captured_regions: Dict):
        """Analyse ultra-réactive de tous les éléments de jeu (4 fois par seconde)"""
        try:
            # 1. ANALYSE DE NOS CARTES (priorité absolue)
            if 'hand_area' in captured_regions:
                my_cards = self.image_analyzer.detect_cards(captured_regions['hand_area'])
                if my_cards:
                    # Vérifier si les cartes ont changé (nouvelle main)
                    if my_cards != self.previous_cards:
                        self.logger.info("Nouvelle main détectée")
                        self._on_new_hand()
                        self.previous_cards = my_cards
                else:
                    self.logger.debug("Aucune carte detectee dans hand_area")
            
            # 2. ANALYSE DES CARTES COMMUNES
            if 'community_cards' in captured_regions:
                community_cards = self.image_analyzer.detect_cards(captured_regions['community_cards'])
                if community_cards:
                    # Cartes communes détectées
                    pass
                else:
                    self.logger.debug("Aucune carte commune detectee")
            
            # 3. ANALYSE DU POT
            if 'pot_area' in captured_regions:
                pot_text = self.image_analyzer.extract_text(captured_regions['pot_area'])
                if pot_text and pot_text.strip():
                    self.logger.info(f"POT: {pot_text}")
                else:
                    self.logger.debug("Aucun pot detecte")
            
            # 4. ANALYSE DE NOTRE STACK
            if 'my_stack_area' in captured_regions:
                stack_text = self.image_analyzer.extract_text(captured_regions['my_stack_area'])
                if stack_text and stack_text.strip():
                    # Stack détecté
                    pass
                else:
                    self.logger.debug("Aucun stack detecte")
            
            # 5. ANALYSE DES STACKS ADVERSAIRES
            for i in range(1, 3):
                stack_key = f'opponent{i}_stack_area'
                if stack_key in captured_regions:
                    stack_text = self.image_analyzer.extract_text(captured_regions[stack_key])
                    if stack_text and stack_text.strip():
                        # Stack adversaire détecté
                        pass
            
            # 6. ANALYSE DES MISES ACTUELLES
            for player in ['my', 'opponent1', 'opponent2']:
                bet_key = f'{player}_current_bet'
                if bet_key in captured_regions:
                    bet_text = self.image_analyzer.extract_text(captured_regions[bet_key])
                    if bet_text and bet_text.strip():
                        self.logger.info(f"{player.upper()} MISE: {bet_text}")
            
            # 7. ANALYSE DES BLINDS
            if 'blinds_area' in captured_regions:
                blinds_text = self.image_analyzer.extract_text(captured_regions['blinds_area'])
                if blinds_text and blinds_text.strip():
                    self.logger.info(f"BLINDS: {blinds_text}")
            
            # 8. ANALYSE DU TIMER
            if 'blinds_timer' in captured_regions:
                timer_text = self.image_analyzer.extract_text(captured_regions['blinds_timer'])
                if timer_text and timer_text.strip():
                    self.logger.info(f"TIMER: {timer_text}")
                    
        except Exception as e:
            self.logger.error(f"Erreur analyse ultra-rapide: {e}")

    def _detect_and_log_cards(self, captured_regions: Dict):
        """Détecte et log les cartes en temps réel"""
        try:
            # 1. Détecter nos cartes
            if 'hand_area' in captured_regions:
                my_cards = self.image_analyzer.detect_cards(captured_regions['hand_area'])
                if my_cards:
                    # Mes cartes détectées
                    pass
                else:
                    self.logger.debug("Aucune carte détectée dans hand_area")
            
            # 2. Détecter les cartes communautaires
            if 'community_cards' in captured_regions:
                community_cards = self.image_analyzer.detect_cards(captured_regions['community_cards'])
                if community_cards:
                    # Cartes communes détectées
                    pass
                else:
                    self.logger.debug("Aucune carte commune détectée")
            
            # 3. Détecter le pot
            if 'pot_area' in captured_regions:
                pot_text = self.image_analyzer.extract_text(captured_regions['pot_area'])
                if pot_text:
                    self.logger.info(f"POT: {pot_text}")
                else:
                    self.logger.debug("💰 Aucun pot détecté")
            
            # 4. Détecter notre stack
            if 'my_stack_area' in captured_regions:
                stack_text = self.image_analyzer.extract_text(captured_regions['my_stack_area'])
                if stack_text:
                    self.logger.info(f"MON STACK: {stack_text}")
                else:
                    self.logger.debug("💰 Aucun stack détecté")
                    
        except Exception as e:
            self.logger.error(f"Erreur détection cartes: {e}")

    def _detect_game_active(self, captured_regions: Dict) -> bool:
        """Détecte si on est en mode "en partie" basé sur les éléments de jeu."""
        try:
            # Vérifier seulement les éléments CRITIQUES pour détecter une partie
            critical_elements = [
                'hand_area',      # Nos cartes (essentiel)
                'pot_area',       # Pot (essentiel)
                'my_stack_area',  # Notre stack (essentiel)
            ]
            
            # Log détaillé pour debug
            self.logger.debug(f"=== DÉTECTION PARTIE ACTIVE ===")
            self.logger.debug(f"Régions capturées: {list(captured_regions.keys())}")
            
            missing_elements = []
            for element in critical_elements:
                if element not in captured_regions:
                    missing_elements.append(element)
                    self.logger.debug(f"❌ {element} - MANQUANT")
                else:
                    img = captured_regions[element]
                    if img is None or img.size == 0:
                        missing_elements.append(element)
                        self.logger.debug(f"❌ {element} - VIDE")
                    else:
                        self.logger.debug(f"✅ {element} - PRÉSENT ({img.shape})")
            
            if missing_elements:
                self.logger.debug(f"Éléments manquants: {missing_elements}")
                return False
            
            # Si les éléments critiques sont présents, on est en partie
            self.logger.info("PARTIE DETECTEE - Elements critiques presents!")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur détection mode 'en partie': {e}")
            return False

    def _continuous_game_analysis_and_decision(self, captured_regions: Dict):
        """Analyse continue et prise de décision ultra-rapide"""
        try:
            # 1. DÉTECTION URGENTE DES BOUTONS D'ACTION
            available_buttons = self._detect_individual_action_buttons(captured_regions)
            
            if available_buttons:
                # Boutons détectés, prise de décision
                
                # 2. ANALYSE RAPIDE DE L'ÉTAT DU JEU
                game_state = self._analyze_complete_game_state(captured_regions)
                
                if game_state:
                    # 3. PRISE DE DÉCISION INSTANTANÉE
                    decision = self._make_instant_decision(game_state, available_buttons)
                    
                    if decision:
                        self.logger.info(f"Décision: {decision['action'].upper()} - {decision['reason']}")
                        
                                    # 4. EXÉCUTION IMMÉDIATE
                        self._execute_action_immediately(decision)
                        
                        # 5. PAUSE COURTE APRÈS ACTION
                        time.sleep(0.5)  # 500ms de pause
                        return True
                    else:
                        self.logger.warning("Aucune décision prise malgré les boutons disponibles")
                        return False
                else:
                    # État de jeu invalide
                    return False
            else:
                # Pas de boutons d'action - continuer l'analyse
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur analyse continue: {e}")
            return False
    
    def _detect_individual_action_buttons(self, captured_regions: Dict) -> List[Dict]:
        """Détecte les boutons d'action individuels"""
        available_buttons = []
        
        # Liste des boutons à vérifier
        button_regions = [
            'fold_button', 'call_button', 'raise_button', 
            'check_button', 'all_in_button'
        ]
        
        for region_name in button_regions:
            if region_name in captured_regions:
                try:
                    # Détecter le bouton spécifique
                    button_img = captured_regions[region_name]
                    if button_img is not None and button_img.size > 0:
                        # Vérifier si le bouton est visible (non vide)
                        if self._is_button_visible(button_img):
                            button_name = region_name.replace('_button', '').upper()
                            available_buttons.append({
                                'name': button_name,
                                'region': region_name,
                                'image': button_img
                            })
                            self.logger.debug(f"Bouton détecté: {button_name}")
                except Exception as e:
                    self.logger.debug(f"Erreur détection {region_name}: {e}")
        
        return available_buttons
    
    def _is_button_visible(self, button_img) -> bool:
        """Vérifie si un bouton est visible (non vide/transparent)"""
        try:
            if button_img is None or button_img.size == 0:
                return False
            
            # Convertir en niveaux de gris
            import cv2
            gray = cv2.cvtColor(button_img, cv2.COLOR_RGB2GRAY)
            
            # Calculer la variance (mesure de contenu)
            variance = cv2.meanStdDev(gray)[1][0][0]
            
            # Si la variance est faible, l'image est probablement vide
            return variance > 10  # Seuil ajustable
            
        except Exception as e:
            self.logger.debug(f"Erreur vérification visibilité: {e}")
            return False
    
    def _analyze_complete_game_state(self, captured_regions: Dict) -> Optional[Dict]:
        """
        Analyse complète de l'état du jeu pour décision Spin & Rush intelligente
        """
        try:
            game_state = {
                'timestamp': time.time(),
                'my_cards': [],
                'community_cards': [],
                'pot_size': 0.0,
                'my_stack': 0.0,
                'opponent_stacks': [],
                'current_bets': [],
                'position': 'BB',  # Par défaut
                'street': 'preflop',
                'big_blind': 1.0,
                'timer': 60,
                'num_players': 3,  # Spin & Rush = 3 joueurs
                'hand_strength': 0.5,
                'spr': 10.0,
                'pot_odds': 0.25
            }
            
            # 1. ANALYSE DE NOS CARTES (CRITIQUE)
            if 'hand_area' in captured_regions:
                my_cards = self.image_analyzer.detect_cards(captured_regions['hand_area'])
                if my_cards:
                    game_state['my_cards'] = [str(card) for card in my_cards]
                    # Cartes détectées
                else:
                    # Aucune carte détectée
                    pass
            
            # 2. ANALYSE DES CARTES COMMUNES
            if 'community_cards' in captured_regions:
                community_cards = self.image_analyzer.detect_cards(captured_regions['community_cards'])
                if community_cards:
                    game_state['community_cards'] = [str(card) for card in community_cards]
                    game_state['street'] = self._determine_street(len(community_cards))
                    self.logger.info(f"Cartes communes: {game_state['community_cards']} ({game_state['street']})")
            
            # 3. ANALYSE DU POT (CRITIQUE)
            if 'pot_area' in captured_regions:
                pot_text = self.image_analyzer.extract_text(captured_regions['pot_area'])
                if pot_text and pot_text.strip():
                    game_state['pot_size'] = self._parse_stack_amount(pot_text)
                    self.logger.info(f"Pot: {pot_text} ({game_state['pot_size']})")
            
            # 4. ANALYSE DE NOTRE STACK (CRITIQUE)
            if 'my_stack_area' in captured_regions:
                stack_text = self.image_analyzer.extract_text(captured_regions['my_stack_area'])
                if stack_text and stack_text.strip():
                    game_state['my_stack'] = self._parse_stack_amount(stack_text)
                    # Stack détecté
                else:
                    game_state['my_stack'] = 500  # Valeur par défaut Spin & Rush
                    self.logger.warning("Stack non detecte, defaut: 500")
            
            # 5. ANALYSE DES STACKS ADVERSAIRES
            for i in range(1, 3):
                stack_key = f'opponent{i}_stack_area'
                if stack_key in captured_regions:
                    stack_text = self.image_analyzer.extract_text(captured_regions[stack_key])
                    if stack_text and stack_text.strip():
                        stack_amount = self._parse_stack_amount(stack_text)
                        game_state['opponent_stacks'].append(stack_amount)
                        # Stack adversaire détecté
            
            # 6. ANALYSE DES MISES ACTUELLES
            for player in ['my', 'opponent1', 'opponent2']:
                bet_key = f'{player}_current_bet'
                if bet_key in captured_regions:
                    bet_text = self.image_analyzer.extract_text(captured_regions[bet_key])
                    if bet_text and bet_text.strip():
                        bet_amount = self._parse_bet_amount(bet_text)
                        game_state['current_bets'].append({
                            'player': player,
                            'amount': bet_amount,
                            'text': bet_text
                        })
                        self.logger.info(f"{player}: {bet_text} ({bet_amount})")
            
            # 7. ANALYSE DES BLINDS (CRITIQUE)
            if 'blinds_area' in captured_regions:
                blinds_text = self.image_analyzer.extract_text(captured_regions['blinds_area'])
                if blinds_text and blinds_text.strip():
                    blinds_info = self._parse_blinds(blinds_text)
                    game_state['big_blind'] = blinds_info.get('big_blind', 1.0)
                    self.logger.info(f"Blinds: {blinds_text} (BB: {game_state['big_blind']})")
            
            # 8. ANALYSE DU TIMER (CRITIQUE POUR SPIN & RUSH)
            if 'blinds_timer' in captured_regions:
                timer_text = self.image_analyzer.extract_text(captured_regions['blinds_timer'])
                if timer_text and timer_text.strip():
                    game_state['timer'] = self._parse_timer(timer_text)
                    
                    # Catégoriser la pression du timer
                    if game_state['timer'] < 15:
                        self.logger.warning("TIMER URGENT! < 15s")
                    elif game_state['timer'] < 30:
                        self.logger.info("Timer sous pression: 15-30s")
                    else:
                        self.logger.info("Timer normal: > 30s")
                    
                    self.logger.info(f"Timer: {timer_text} ({game_state['timer']}s)")
                else:
                    game_state['timer'] = 60  # Valeur par défaut
                    self.logger.warning("Timer non detecte, defaut: 60s")
            
            # 9. DÉTERMINATION DE LA POSITION
            game_state['position'] = self._get_current_position()
            
            # 10. CALCULS AVANCÉS POUR SPIN & RUSH
            if game_state['my_stack'] > 0 and game_state['pot_size'] > 0:
                spr = self._calculate_stack_to_pot_ratio(game_state['my_stack'], game_state['pot_size'])
                game_state['spr'] = spr
                
                # Calculer le montant à payer pour call
                call_amount = self._get_call_amount(game_state['current_bets'])
                if call_amount > 0:
                    pot_odds = self._calculate_pot_odds(game_state['pot_size'], call_amount)
                    game_state['pot_odds'] = pot_odds
                    game_state['call_amount'] = call_amount
            
            # 11. ÉVALUATION DE LA FORCE DE MAIN
            if game_state['my_cards']:
                hand_strength = self._calculate_hand_strength_ultra_fast(
                    game_state['my_cards'], 
                    game_state['community_cards']
                )
                game_state['hand_strength'] = hand_strength
                self.logger.info(f"Force de main: {hand_strength:.2f}")
            
            # 12. VALIDATION CRITIQUE
            if self._validate_game_data_quick(game_state):
                # État du jeu analysé
                return game_state
            else:
                # Données de jeu invalides
                return None
                
        except Exception as e:
            self.logger.error(f"Erreur analyse état Spin & Rush: {e}")
            return None
    
    def _strategic_thinking(self, game_state: Dict):
        """Réflexion stratégique en arrière-plan"""
        try:
            # Calculs avancés pour la prise de décision
            if game_state['my_cards']:
                # Force de main
                hand_strength = self._quick_hand_evaluation(
                    game_state['my_cards'], 
                    game_state['community_cards']
                )
                
                # Stack-to-Pot Ratio
                spr = self._calculate_stack_to_pot_ratio(
                    game_state['my_stack'], 
                    game_state['pot']
                )
                
                # Pot odds
                call_amount = self._get_call_amount(game_state['current_bets'])
                pot_odds = self._calculate_pot_odds(game_state['pot'], call_amount)
                
                # Stocker pour décision instantanée
                self.current_strategy = {
                    'hand_strength': hand_strength,
                    'spr': spr,
                    'pot_odds': pot_odds,
                    'position': game_state['position'],
                    'street': game_state['street'],
                    'call_amount': call_amount
                }
                
                self.logger.debug(f"STRATÉGIE: Force={hand_strength:.2f}, SPR={spr:.2f}, Pot Odds={pot_odds:.2f}")
                
        except Exception as e:
            self.logger.error(f"Erreur réflexion stratégique: {e}")
    
    def _make_instant_decision(self, game_state: Dict, available_buttons: List[Dict]) -> Dict:
        """
        Décision ultra-rapide avec IA avancée intégrée
        """
        start_time = time.time()
        
        try:
            # NOUVEAU: Vérifier le cache de décision
            decision_key = self._generate_decision_key(game_state, available_buttons)
            if decision_key in self.decision_cache:
                cached_decision = self.decision_cache[decision_key]
                if time.time() - cached_decision['timestamp'] < 0.1:  # 100ms TTL
                    self.logger.debug("Cache hit pour décision")
                    return cached_decision['decision']
            
            # Extraire les informations critiques du jeu
            my_cards = game_state.get('my_cards', []) if game_state else []
            community_cards = game_state.get('community_cards', []) if game_state else []
            my_stack = game_state.get('my_stack', 0) if game_state else 500
            pot_size = game_state.get('pot_size', 0) if game_state else 0
            big_blind = game_state.get('big_blind', 1) if game_state else 1
            position = game_state.get('position', 'BB') if game_state else 'BB'
            street = game_state.get('street', 'preflop') if game_state else 'preflop'
            timer = game_state.get('timer', 60) if game_state else 60
            num_players = game_state.get('num_players', 3) if game_state else 3
            
            # NOUVEAU: Validation des cartes avec couleurs
            valid_cards = []
            for card in my_cards:
                if isinstance(card, str) and len(card) >= 2:
                    rank = card[0]
                    suit = card[1] if len(card) > 1 else '?'
                    if suit in '♠♥♦♣':
                        valid_cards.append(card)
            
            if not valid_cards:
                self.logger.warning("Aucune carte valide détectée")
                valid_cards = my_cards  # Utiliser les cartes même sans couleur
            
            # NOUVEAU: Calcul avancé de force de main
            hand_strength = self._calculate_advanced_hand_strength(valid_cards, community_cards)
            
            # NOUVEAU: Calculs avancés avec IA
            spr = self._calculate_stack_to_pot_ratio(my_stack, pot_size) if pot_size > 0 else 10
            pot_odds = self._calculate_pot_odds(pot_size, big_blind) if big_blind > 0 else 0.25
            equity = self._calculate_equity_vs_range(valid_cards, community_cards)
            
            # NOUVEAU: Métriques avancées
            position_bonus = self._get_position_bonus(position)
            timer_pressure = self._get_timer_pressure(timer)
            stack_pressure = self._get_stack_pressure(my_stack, pot_size)
            
            # Validation de l'action selon les boutons disponibles
            available_actions = [btn['name'].lower() for btn in available_buttons]
            
            # NOUVEAU: Décision avec IA avancée
            decision = self._make_ai_decision(
                valid_cards, community_cards, hand_strength, spr, pot_odds, equity,
                position_bonus, timer_pressure, stack_pressure, available_actions, timer
            )
            
            # NOUVEAU: Cache de la décision
            if decision:
                self.decision_cache[decision_key] = {
                    'decision': decision,
                    'timestamp': time.time()
                }
                
                # Limiter la taille du cache
                if len(self.decision_cache) > 1000:
                    oldest_key = min(self.decision_cache.keys(), 
                                   key=lambda k: self.decision_cache[k]['timestamp'])
                    del self.decision_cache[oldest_key]
            
            # Métriques de performance
            decision_time = time.time() - start_time
            self.performance_metrics['decision_times'].append(decision_time)
            if len(self.performance_metrics['decision_times']) > 100:
                self.performance_metrics['decision_times'].pop(0)
            
            self.session_stats['decisions_made'] += 1
            
            return decision if decision else {'action': 'fold', 'reason': 'Fallback'}

        except Exception as e:
            self.logger.error(f"Erreur décision IA: {e}")
            self.session_stats['errors_count'] += 1
            # Fallback: fold si possible, sinon check
            available_actions = [btn['name'].lower() for btn in available_buttons]
            if 'fold' in available_actions:
                return {'action': 'fold', 'reason': 'Erreur'}
            elif 'check' in available_actions:
                return {'action': 'check', 'reason': 'Erreur'}
            else:
                return {'action': 'fold', 'reason': 'Erreur'}

    def _generate_decision_key(self, game_state: Dict, available_buttons: List[Dict]) -> str:
        """
        Génère une clé unique pour le cache de décision
        """
        try:
            my_cards = game_state.get('my_cards', []) if game_state else []
            community_cards = game_state.get('community_cards', []) if game_state else []
            my_stack = game_state.get('my_stack', 0) if game_state else 500
            pot_size = game_state.get('pot_size', 0) if game_state else 0
            position = game_state.get('position', 'BB') if game_state else 'BB'
            timer = game_state.get('timer', 60) if game_state else 60
            
            button_names = [btn['name'] for btn in available_buttons]
            
            key_parts = [
                str(sorted(my_cards)),
                str(sorted(community_cards)),
                f"{my_stack:.1f}",
                f"{pot_size:.1f}",
                position,
                f"{timer}",
                str(sorted(button_names))
            ]
            
            return "|".join(key_parts)
            
        except Exception as e:
            self.logger.debug(f"Erreur génération clé: {e}")
            return str(time.time())

    def _calculate_advanced_hand_strength(self, my_cards: List[str], community_cards: List[str]) -> float:
        """
        Calcul avancé de force de main avec IA
        """
        try:
            if not my_cards:
                return 0.0
            
            # NOUVEAU: Utiliser l'IA avancée si disponible
            if hasattr(self, 'advanced_ai') and self.advanced_ai:
                try:
                    return self.advanced_ai.calculate_hand_strength(my_cards, community_cards)
                except Exception as e:
                    self.logger.debug(f"IA avancée échouée, fallback: {e}")
            
            # Fallback vers le calcul rapide
            return self._calculate_hand_strength_ultra_fast(my_cards, community_cards)
            
        except Exception as e:
            self.logger.error(f"Erreur calcul force main: {e}")
            return 0.0

    def _make_ai_decision(self, my_cards: List[str], community_cards: List[str], 
                         hand_strength: float, spr: float, pot_odds: float, equity: float,
                         position_bonus: float, timer_pressure: float, stack_pressure: float,
                         available_actions: List[str], timer: int) -> Dict:
        """
        Décision intelligente avec IA avancée
        """
        try:
            # NOUVEAU: Score composite
            composite_score = (
                hand_strength * 0.4 +
                equity * 0.3 +
                position_bonus * 0.1 +
                (1 - timer_pressure) * 0.1 +
                (1 - stack_pressure) * 0.1
            )
            
            # NOUVEAU: Logique adaptative
            if timer < 10:  # TIMER CRITIQUE
                if 'all_in' in available_actions and composite_score > 0.3:
                    return {'action': 'all_in', 'reason': 'Timer critique'}
                elif 'raise' in available_actions and composite_score > 0.2:
                    return {'action': 'raise', 'reason': 'Timer critique'}
                elif 'call' in available_actions and composite_score > 0.1:
                    return {'action': 'call', 'reason': 'Timer critique'}
            
            # NOUVEAU: Stratégie basée sur le score composite
            if composite_score > 0.8:  # Main excellente
                if 'raise' in available_actions:
                    return {'action': 'raise', 'reason': 'Main excellente'}
                elif 'call' in available_actions:
                    return {'action': 'call', 'reason': 'Main excellente'}
            elif composite_score > 0.6:  # Main forte
                if 'raise' in available_actions:
                    return {'action': 'raise', 'reason': 'Main forte'}
                elif 'call' in available_actions:
                    return {'action': 'call', 'reason': 'Main forte'}
            elif composite_score > 0.4:  # Main moyenne
                if 'call' in available_actions and pot_odds > 0.3:
                    return {'action': 'call', 'reason': 'Pot odds favorables'}
                elif 'check' in available_actions:
                    return {'action': 'check', 'reason': 'Main moyenne'}
            elif composite_score > 0.2:  # Main faible
                if 'check' in available_actions:
                    return {'action': 'check', 'reason': 'Main faible'}
                elif 'fold' in available_actions:
                    return {'action': 'fold', 'reason': 'Main faible'}
            else:  # Main très faible
                if 'fold' in available_actions:
                    return {'action': 'fold', 'reason': 'Main très faible'}
                elif 'check' in available_actions:
                    return {'action': 'check', 'reason': 'Main très faible'}
            
            # Fallback
            if 'fold' in available_actions:
                return {'action': 'fold', 'reason': 'Fallback'}
            elif 'check' in available_actions:
                return {'action': 'check', 'reason': 'Fallback'}
            else:
                return {'action': 'fold', 'reason': 'Fallback'}
                
        except Exception as e:
            self.logger.error(f"Erreur décision IA: {e}")
            return {'action': 'fold', 'reason': 'Erreur'}

    def _get_position_bonus(self, position: str) -> float:
        """Bonus de position"""
        position_bonuses = {
            'BTN': 0.3, 'CO': 0.2, 'MP': 0.1, 'UTG': 0.0, 'BB': 0.1, 'SB': 0.0
        }
        return position_bonuses.get(position, 0.0)

    def _get_timer_pressure(self, timer: int) -> float:
        """Pression du timer (0-1)"""
        if timer < 10:
            return 1.0
        elif timer < 20:
            return 0.8
        elif timer < 30:
            return 0.5
        else:
            return 0.0

    def _get_stack_pressure(self, stack: float, pot: float) -> float:
        """Pression du stack (0-1)"""
        if pot == 0:
            return 0.0
        spr = stack / pot
        if spr < 1:
            return 1.0
        elif spr < 3:
            return 0.7
        elif spr < 10:
            return 0.3
        else:
            return 0.0
    
    def _determine_street(self, community_cards_count: int) -> str:
        """Détermine la rue actuelle"""
        if community_cards_count == 0:
            return 'preflop'
        elif community_cards_count == 3:
            return 'flop'
        elif community_cards_count == 4:
            return 'turn'
        elif community_cards_count == 5:
            return 'river'
        else:
            return 'unknown'
    
    def _get_call_amount(self, current_bets: List[Dict]) -> float:
        """Calcule le montant à payer pour call"""
        if not current_bets:
            return 0.0
        
        # Trouver la mise la plus élevée
        max_bet = max(bet['amount'] for bet in current_bets)
        my_bet = next((bet['amount'] for bet in current_bets if bet['player'] == 'my'), 0.0)
        
        return max_bet - my_bet
    
    def _parse_blinds(self, blinds_text: str) -> Dict:
        """Parse les blinds"""
        try:
            # Format attendu: "10/20" ou "SB/BB"
            parts = blinds_text.split('/')
            if len(parts) == 2:
                return {
                    'small_blind': self._parse_stack_amount(parts[0]),
                    'big_blind': self._parse_stack_amount(parts[1])
                }
            return {'small_blind': 0, 'big_blind': 0}
        except:
            return {'small_blind': 0, 'big_blind': 0}
    
    def _detect_winner_crown(self, captured_regions: Dict) -> bool:
        """
        Détecte la couronne de victoire (winner2.png) dans l'écran complet
        Retourne True si la couronne est détectée (fin de manche)
        """
        try:
            # Capturer l'écran complet pour détecter la couronne
            full_screenshot = self.screen_capture.capture_full_screen()
            if full_screenshot is None:
                return False
            
            # Utiliser le ButtonDetector pour détecter la couronne
            crown_detected = self.button_detector.detect_winner_crown(full_screenshot)
            
            if crown_detected:
                self.logger.info("🎉 COURONNE DE VICTOIRE DÉTECTÉE - Fin de manche!")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Erreur détection couronne de victoire: {e}")
            return False

    def _safe_execute(self, func, *args, **kwargs):
        """
        Exécution sécurisée avec gestion d'erreurs et recovery
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Erreur dans {func.__name__}: {e}")
            self.session_stats['errors_count'] += 1
            return None

    def _recover_from_error(self, error_type: str, context: Dict = None):
        """
        Recovery automatique après erreur
        """
        try:
            self.logger.warning(f"Recovery automatique pour: {error_type}")
            
            if error_type == 'capture_error':
                # Réinitialiser le cache
                self.image_cache.clear()
                self.last_capture_time = 0
                time.sleep(0.1)
                
            elif error_type == 'decision_error':
                # Réinitialiser le cache de décision
                self.decision_cache.clear()
                time.sleep(0.05)
                
            elif error_type == 'ocr_error':
                # Réinitialiser l'analyseur d'image
                self.image_analyzer = ImageAnalyzer()
                time.sleep(0.1)
                
            elif error_type == 'button_error':
                # Réinitialiser le détecteur de boutons
                self.button_detector = ButtonDetector()
                time.sleep(0.05)
                
            self.logger.info(f"Recovery terminé pour: {error_type}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du recovery: {e}")

    def _validate_game_data(self, game_data: Dict) -> bool:
        """
        Validation robuste des données de jeu
        """
        try:
            if not game_data:
                return False
            
            # Vérifications de base
            required_fields = ['my_cards', 'my_stack', 'pot_size', 'position', 'timer']
            for field in required_fields:
                if field not in game_data:
                    self.logger.debug(f"Champ manquant: {field}")
                    return False
            
            # Validation des cartes
            my_cards = game_data.get('my_cards', [])
            if not isinstance(my_cards, list):
                return False
            
            # Validation du stack
            my_stack = game_data.get('my_stack', 0)
            if not isinstance(my_stack, (int, float)) or my_stack < 0:
                return False
            
            # Validation du pot
            pot_size = game_data.get('pot_size', 0)
            if not isinstance(pot_size, (int, float)) or pot_size < 0:
                return False
            
            # Validation du timer
            timer = game_data.get('timer', 0)
            if not isinstance(timer, (int, float)) or timer < 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur validation données: {e}")
            return False

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