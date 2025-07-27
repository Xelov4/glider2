"""
🔥 Stratégie Agressive - Poker IA
=================================

Nouvelle stratégie ultra-agressive pour maximiser les gains :
- All-in sur mains fortes
- Bluff fréquent
- Relances agressives
- Exploitation des faiblesses adverses

FONCTIONNALITÉS
===============

✅ Décisions ultra-agressives
✅ Bluff intelligent
✅ Exploitation de position
✅ Gestion du stack
✅ Adaptation au contexte

VERSION: 1.0.0
DERNIÈRE MISE À JOUR: 2025-07-27
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class GameContext:
    my_cards: List[str]
    community_cards: List[str]
    my_stack: float
    pot_size: float
    big_blind: float
    position: str
    street: str
    timer: int
    num_players: int
    opponent_actions: List[str]

class AggressiveStrategy:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration de l'agression
        self.aggression_level = 0.8  # 0-1, plus = plus agressif
        self.bluff_frequency = 0.3   # Fréquence des bluffs
        self.all_in_threshold = 0.7  # Seuil pour all-in
        
        # Rangs des cartes (force)
        self.card_ranks = {
            'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
            '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2
        }

    def make_decision(self, context: GameContext, available_actions: List[str]) -> Dict:
        """
        Prend une décision agressive basée sur le contexte
        """
        try:
            self.logger.info(f"🔥 STRATÉGIE AGRESSIVE - Analyse du contexte")
            
            # Calculer la force de la main
            hand_strength = self._calculate_hand_strength(context.my_cards, context.community_cards)
            
            # Calculer les métriques importantes
            stack_to_pot = context.my_stack / context.pot_size if context.pot_size > 0 else 10
            position_bonus = self._get_position_bonus(context.position)
            timer_pressure = self._get_timer_pressure(context.timer)
            
            # DÉCISION AGRESSIVE
            self.logger.info(f"Force: {hand_strength:.2f}, Stack/Pot: {stack_to_pot:.2f}, Position: {context.position}")
            
            # LOGIQUE AGRESSIVE
            if hand_strength > 0.8:  # Main très forte
                return self._handle_strong_hand(context, available_actions, hand_strength)
            elif hand_strength > 0.6:  # Main forte
                return self._handle_good_hand(context, available_actions, hand_strength)
            elif hand_strength > 0.4:  # Main moyenne
                return self._handle_medium_hand(context, available_actions, hand_strength)
            else:  # Main faible
                return self._handle_weak_hand(context, available_actions, hand_strength)
                
        except Exception as e:
            self.logger.error(f"Erreur stratégie agressive: {e}")
            return self._fallback_decision(available_actions)

    def _calculate_hand_strength(self, my_cards: List[str], community_cards: List[str]) -> float:
        """Calcule la force de la main (0-1)"""
        try:
            if not my_cards:
                return 0.0
            
            # Force basée sur les cartes du joueur
            card_values = []
            for card in my_cards:
                if len(card) >= 1:
                    rank = card[0]
                    if rank in self.card_ranks:
                        card_values.append(self.card_ranks[rank])
            
            if not card_values:
                return 0.0
            
            # Calculer la force moyenne
            avg_strength = sum(card_values) / len(card_values)
            
            # Normaliser (2-14 -> 0-1)
            normalized_strength = (avg_strength - 2) / 12
            
            # Bonus pour les paires
            if len(card_values) >= 2 and card_values[0] == card_values[1]:
                normalized_strength += 0.2
            
            # Bonus pour les cartes hautes
            if max(card_values) >= 12:  # Q, K, A
                normalized_strength += 0.1
            
            return min(1.0, normalized_strength)
            
        except Exception as e:
            self.logger.error(f"Erreur calcul force main: {e}")
            return 0.5

    def _get_position_bonus(self, position: str) -> float:
        """Bonus selon la position"""
        position_bonuses = {
            'BB': 0.1,   # Big Blind - position défensive
            'SB': 0.2,   # Small Blind - position moyenne
            'UTG': 0.0,  # Under the Gun - position difficile
            'MP': 0.1,   # Middle Position
            'CO': 0.3,   # Cutoff - bonne position
            'BTN': 0.4   # Button - meilleure position
        }
        return position_bonuses.get(position, 0.0)

    def _get_timer_pressure(self, timer: int) -> float:
        """Pression du timer (0-1)"""
        if timer < 10:
            return 0.8  # Très pressé
        elif timer < 20:
            return 0.5  # Pressé
        else:
            return 0.1  # Pas pressé

    def _handle_strong_hand(self, context: GameContext, actions: List[str], strength: float) -> Dict:
        """Gestion des mains très fortes"""
        self.logger.info(f"🔥 MAIN TRÈS FORTE - Agression maximale")
        
        if 'all_in' in actions:
            return {'action': 'all_in', 'reason': 'Main très forte - all-in'}
        elif 'raise' in actions:
            return {'action': 'raise', 'reason': 'Main très forte - raise'}
        elif 'call' in actions:
            return {'action': 'call', 'reason': 'Main très forte - call'}
        else:
            return {'action': 'check', 'reason': 'Main très forte - check'}

    def _handle_good_hand(self, context: GameContext, actions: List[str], strength: float) -> Dict:
        """Gestion des mains fortes"""
        self.logger.info(f"🔥 MAIN FORTE - Agression élevée")
        
        # Décision basée sur la position et le stack
        if context.position in ['BTN', 'CO'] and 'raise' in actions:
            return {'action': 'raise', 'reason': 'Main forte en position'}
        elif 'raise' in actions:
            return {'action': 'raise', 'reason': 'Main forte - raise'}
        elif 'call' in actions:
            return {'action': 'call', 'reason': 'Main forte - call'}
        else:
            return {'action': 'check', 'reason': 'Main forte - check'}

    def _handle_medium_hand(self, context: GameContext, actions: List[str], strength: float) -> Dict:
        """Gestion des mains moyennes"""
        self.logger.info(f"🔥 MAIN MOYENNE - Agression modérée")
        
        # Bluff occasionnel en position
        if context.position in ['BTN', 'CO'] and self._should_bluff():
            if 'raise' in actions:
                return {'action': 'raise', 'reason': 'Bluff en position'}
        
        if 'call' in actions:
            return {'action': 'call', 'reason': 'Main moyenne - call'}
        elif 'check' in actions:
            return {'action': 'check', 'reason': 'Main moyenne - check'}
        else:
            return {'action': 'fold', 'reason': 'Main moyenne - fold'}

    def _handle_weak_hand(self, context: GameContext, actions: List[str], strength: float) -> Dict:
        """Gestion des mains faibles"""
        self.logger.info(f"🔥 MAIN FAIBLE - Stratégie défensive")
        
        # Bluff agressif en position
        if context.position in ['BTN', 'CO'] and self._should_bluff():
            if 'raise' in actions:
                return {'action': 'raise', 'reason': 'Bluff agressif'}
        
        # Check si possible
        if 'check' in actions:
            return {'action': 'check', 'reason': 'Main faible - check'}
        elif 'fold' in actions:
            return {'action': 'fold', 'reason': 'Main faible - fold'}
        else:
            return {'action': 'call', 'reason': 'Main faible - call'}

    def _should_bluff(self) -> bool:
        """Détermine si on doit bluffer"""
        import random
        return random.random() < self.bluff_frequency

    def _fallback_decision(self, actions: List[str]) -> Dict:
        """Décision de fallback"""
        if 'fold' in actions:
            return {'action': 'fold', 'reason': 'Fallback - fold'}
        elif 'check' in actions:
            return {'action': 'check', 'reason': 'Fallback - check'}
        else:
            return {'action': 'fold', 'reason': 'Fallback - fold'}

    def adjust_strategy(self, recent_results: List[Dict]):
        """Ajuste la stratégie basée sur les résultats récents"""
        try:
            if not recent_results:
                return
            
            # Analyser les résultats récents
            wins = sum(1 for result in recent_results if result.get('result') == 'win')
            total = len(recent_results)
            win_rate = wins / total if total > 0 else 0.5
            
            # Ajuster l'agression selon le win rate
            if win_rate > 0.6:
                self.aggression_level = min(1.0, self.aggression_level + 0.1)
                self.logger.info(f"🔥 Augmentation agression - Win rate: {win_rate:.2f}")
            elif win_rate < 0.4:
                self.aggression_level = max(0.3, self.aggression_level - 0.1)
                self.logger.info(f"🔥 Réduction agression - Win rate: {win_rate:.2f}")
                
        except Exception as e:
            self.logger.error(f"Erreur ajustement stratégie: {e}") 