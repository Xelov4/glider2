"""
Module de décision IA pour l'agent IA Poker
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from .game_state import GameState, Action, Position

@dataclass
class Decision:
    """Représente une décision de l'IA"""
    action: str
    amount: int = 0
    confidence: float = 0.0
    reasoning: str = ""
    ev: float = 0.0

class OpponentModel:
    """
    Modèle d'un adversaire pour exploitative play
    """
    
    def __init__(self, opponent_id: str):
        self.opponent_id = opponent_id
        self.actions_history = []
        self.vpip = 0.0  # Voluntarily Put Money In Pot
        self.pfr = 0.0   # Pre-Flop Raise
        self.af = 0.0    # Aggression Factor
        self.hands_played = 0
        self.total_actions = 0
        
    def update_model(self, action: Action, state: GameState):
        """
        Met à jour le modèle avec une nouvelle action
        """
        self.actions_history.append({
            'action': action,
            'street': state.street,
            'pot_size': state.pot_size,
            'position': state.my_position.value
        })
        
        self.total_actions += 1
        self.hands_played = len(set([a['street'] for a in self.actions_history]))
        
        # Calcul des statistiques
        self.calculate_statistics()
    
    def calculate_statistics(self):
        """
        Calcule les statistiques de l'adversaire
        """
        if not self.actions_history:
            return
        
        # VPIP (Voluntarily Put Money In Pot)
        voluntary_actions = [a for a in self.actions_history 
                           if a['action'] in [Action.CALL, Action.BET, Action.RAISE]]
        self.vpip = len(voluntary_actions) / max(1, self.total_actions)
        
        # PFR (Pre-Flop Raise)
        preflop_raises = [a for a in self.actions_history 
                         if a['street'] == 'preflop' and a['action'] == Action.RAISE]
        preflop_actions = [a for a in self.actions_history if a['street'] == 'preflop']
        self.pfr = len(preflop_raises) / max(1, len(preflop_actions))
        
        # AF (Aggression Factor)
        aggressive_actions = [a for a in self.actions_history 
                           if a['action'] in [Action.BET, Action.RAISE]]
        passive_actions = [a for a in self.actions_history 
                         if a['action'] in [Action.CALL, Action.CHECK]]
        self.af = len(aggressive_actions) / max(1, len(passive_actions))

class AIDecisionMaker:
    """
    Module de prise de décision IA basé sur la théorie des jeux
    """
    
    def __init__(self):
        self.opponent_models = {}
        self.risk_tolerance = 0.8
        self.aggression_level = 0.7
        self.bluff_frequency = 0.15
        self.logger = logging.getLogger(__name__)
        
        # Paramètres de stratégie
        self.gto_weight = 0.6
        self.exploitative_weight = 0.4
        
    def make_decision(self, state: GameState) -> Decision:
        """
        Prend une décision basée sur l'état du jeu
        """
        try:
            if not state.is_my_turn:
                return Decision(action='wait', confidence=1.0, reasoning="Pas notre tour")
            
            # Évaluation de la situation
            hand_strength = self.poker_engine.evaluate_hand_strength(state.my_cards, state.community_cards)
            pot_odds = self.poker_engine.calculate_pot_odds(state.pot_size, state.current_bet)
            
            # Simulation Monte Carlo
            simulation_results = self.poker_engine.simulate_outcomes(state, num_simulations=500)
            
            # Calcul des EVs pour chaque action
            ev_fold = 0.0
            ev_call = simulation_results['ev_call']
            ev_raise = simulation_results['ev_raise']
            
            # Décision basée sur les EVs
            best_action = 'fold'
            best_ev = ev_fold
            best_amount = 0
            
            if ev_call > best_ev:
                best_action = 'call'
                best_ev = ev_call
                best_amount = state.current_bet
            
            if ev_raise > best_ev:
                best_action = 'raise'
                best_ev = ev_raise
                best_amount = self.poker_engine.calculate_bet_sizing(state, 'raise')
            
            # Ajustements stratégiques
            decision = self.apply_strategic_adjustments(
                state, best_action, best_ev, best_amount, hand_strength, simulation_results
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Erreur prise de décision: {e}")
            return Decision(action='fold', confidence=0.0, reasoning="Erreur système")
    
    def apply_strategic_adjustments(self, state: GameState, action: str, ev: float, 
                                  amount: int, hand_strength: float, 
                                  simulation_results: Dict) -> Decision:
        """
        Applique les ajustements stratégiques à la décision
        """
        # Décision GTO de base
        gto_decision = Decision(
            action=action,
            amount=amount,
            confidence=min(1.0, abs(ev) / 100.0),
            reasoning=f"GTO: EV={ev:.2f}, Win rate={simulation_results['win_rate']:.2%}",
            ev=ev
        )
        
        # Ajustements exploitatifs
        exploitative_decision = self.apply_exploitative_adjustments(
            state, gto_decision, hand_strength
        )
        
        # Ajustements de bluff
        bluff_decision = self.apply_bluff_adjustments(
            state, exploitative_decision, hand_strength
        )
        
        # Ajustements de position
        position_decision = self.apply_position_adjustments(
            state, bluff_decision
        )
        
        return position_decision
    
    def apply_exploitative_adjustments(self, state: GameState, decision: Decision, 
                                     hand_strength: float) -> Decision:
        """
        Applique les ajustements exploitatifs basés sur les modèles d'adversaires
        """
        if not state.players:
            return decision
        
        # Analyser les adversaires actifs
        active_opponents = [p for p in state.players if p.is_active and p.id != 'me']
        
        if not active_opponents:
            return decision
        
        # Calculer la tendance moyenne des adversaires
        avg_vpip = np.mean([self.get_opponent_model(opp.id).vpip for opp in active_opponents])
        avg_af = np.mean([self.get_opponent_model(opp.id).af for opp in active_opponents])
        
        # Ajustements basés sur les tendances
        if avg_vpip > 0.3:  # Adversaires loose
            if decision.action == 'call' and hand_strength > 0.4:
                decision.action = 'raise'
                decision.amount = int(decision.amount * 1.2)
                decision.reasoning += " | Exploitative: Loose opponents"
        
        if avg_af < 1.0:  # Adversaires passifs
            if decision.action == 'call' and hand_strength > 0.3:
                decision.action = 'raise'
                decision.amount = int(decision.amount * 1.3)
                decision.reasoning += " | Exploitative: Passive opponents"
        
        return decision
    
    def apply_bluff_adjustments(self, state: GameState, decision: Decision, 
                              hand_strength: float) -> Decision:
        """
        Applique les ajustements de bluff
        """
        # Décider si on doit bluffer
        should_bluff = self.should_bluff(state)
        
        if should_bluff and hand_strength < 0.3:
            # Bluff avec une main faible
            if decision.action == 'fold':
                decision.action = 'raise'
                decision.amount = int(state.pot_size * 0.5)
                decision.confidence = 0.6
                decision.reasoning += " | Bluff"
        
        return decision
    
    def apply_position_adjustments(self, state: GameState, decision: Decision) -> Decision:
        """
        Applique les ajustements basés sur la position
        """
        if state.my_position == Position.LATE:
            # Position tardive - plus agressif
            if decision.action == 'call' and decision.confidence > 0.5:
                decision.action = 'raise'
                decision.amount = int(decision.amount * 1.1)
                decision.reasoning += " | Late position"
        
        elif state.my_position == Position.EARLY:
            # Position précoce - plus conservateur
            if decision.action == 'raise' and decision.confidence < 0.7:
                decision.action = 'call'
                decision.amount = state.current_bet
                decision.reasoning += " | Early position"
        
        return decision
    
    def should_bluff(self, state: GameState) -> bool:
        """
        Détermine s'il faut bluffer
        """
        # Facteurs pour le bluff
        bluff_factors = []
        
        # Position tardive
        if state.my_position == Position.LATE:
            bluff_factors.append(0.3)
        
        # Peu d'adversaires
        active_opponents = len([p for p in state.players if p.is_active])
        if active_opponents <= 2:
            bluff_factors.append(0.2)
        
        # Pot relativement petit
        if state.pot_size < state.my_stack * 0.1:
            bluff_factors.append(0.2)
        
        # Probabilité de bluff basée sur les facteurs
        bluff_probability = min(0.8, sum(bluff_factors))
        
        return random.random() < bluff_probability
    
    def update_opponent_model(self, opponent_id: str, action: Action, state: GameState):
        """
        Met à jour le modèle d'un adversaire
        """
        if opponent_id not in self.opponent_models:
            self.opponent_models[opponent_id] = OpponentModel(opponent_id)
        
        self.opponent_models[opponent_id].update_model(action, state)
    
    def get_opponent_model(self, opponent_id: str) -> OpponentModel:
        """
        Récupère le modèle d'un adversaire
        """
        if opponent_id not in self.opponent_models:
            self.opponent_models[opponent_id] = OpponentModel(opponent_id)
        
        return self.opponent_models[opponent_id]
    
    def calculate_ev(self, action: str, state: GameState) -> float:
        """
        Calcule l'espérance de gain d'une action
        """
        try:
            if action == 'fold':
                return 0.0
            
            # Simulation pour calculer l'EV
            simulation_results = self.poker_engine.simulate_outcomes(state, num_simulations=200)
            
            if action == 'call':
                return simulation_results['ev_call']
            elif action == 'raise':
                return simulation_results['ev_raise']
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Erreur calcul EV: {e}")
            return 0.0
    
    def get_optimal_bet_size(self, state: GameState, action: str) -> int:
        """
        Calcule la taille de mise optimale
        """
        try:
            if action not in ['bet', 'raise']:
                return 0
            
            # Taille basée sur la force de la main
            hand_strength = self.poker_engine.evaluate_hand_strength(state.my_cards, state.community_cards)
            
            if action == 'bet':
                if hand_strength > 0.8:  # Main très forte
                    return int(state.pot_size * 0.75)
                elif hand_strength > 0.6:  # Main forte
                    return int(state.pot_size * 0.5)
                else:  # Main faible
                    return int(state.pot_size * 0.25)
            
            elif action == 'raise':
                current_bet = state.current_bet
                if current_bet == 0:
                    return int(state.pot_size * 0.5)
                else:
                    return current_bet * 2
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Erreur calcul taille mise: {e}")
            return 0
    
    def should_continue_hand(self, state: GameState) -> bool:
        """
        Détermine si on doit continuer la main
        """
        try:
            # Facteurs pour continuer
            hand_strength = self.poker_engine.evaluate_hand_strength(state.my_cards, state.community_cards)
            pot_odds = self.poker_engine.calculate_pot_odds(state.pot_size, state.current_bet)
            
            # Règle de base: continuer si hand_strength > pot_odds
            if hand_strength > pot_odds:
                return True
            
            # Ajustements basés sur la position
            if state.my_position == Position.LATE:
                return hand_strength > pot_odds * 0.8  # Plus loose en position tardive
            
            return False
            
        except Exception as e:
            self.logger.error(f"Erreur décision continuer: {e}")
            return False
    
    def set_risk_tolerance(self, tolerance: float):
        """
        Définit la tolérance au risque (0-1)
        """
        self.risk_tolerance = max(0.0, min(1.0, tolerance))
    
    def set_aggression_level(self, level: float):
        """
        Définit le niveau d'agressivité (0-1)
        """
        self.aggression_level = max(0.0, min(1.0, level))
    
    def set_bluff_frequency(self, frequency: float):
        """
        Définit la fréquence de bluff (0-1)
        """
        self.bluff_frequency = max(0.0, min(1.0, frequency)) 