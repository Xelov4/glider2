"""
Module d'automatisation pour l'agent IA Poker
"""

import time
import random
import pyautogui
import keyboard
import mouse
from typing import Tuple, Optional
import logging
import math
from pynput import mouse as pynput_mouse
from pynput import keyboard as pynput_keyboard

class AutomationEngine:
    """
    Module de contrôle d'automatisation avec anti-détection
    """
    
    def __init__(self):
        self.click_randomization = 5  # pixels
        self.move_speed_range = (0.1, 0.3)  # secondes
        self.human_delays = True
        self.logger = logging.getLogger(__name__)
        
        # Configuration anti-détection
        self.last_action_time = 0
        self.min_delay_between_actions = 0.5
        self.max_delay_between_actions = 2.0
        
        # Historique des mouvements pour éviter les patterns
        self.movement_history = []
        self.max_history_size = 10
        
        # Configuration des mouvements courbes
        self.curve_intensity = 0.3
        
        # Désactiver le fail-safe de PyAutoGUI
        pyautogui.FAILSAFE = False
        
    def click_button(self, button_name: str, coordinates: Tuple[int, int]):
        """
        Clique sur un bouton avec mouvement naturel
        """
        try:
            # Vérifier le délai minimum
            self.ensure_minimum_delay()
            
            # Coordonnées avec randomisation
            x, y = coordinates
            x += random.randint(-self.click_randomization, self.click_randomization)
            y += random.randint(-self.click_randomization, self.click_randomization)
            
            # Mouvement courbe vers la cible
            self.move_to_coordinates_curved((x, y))
            
            # Délai humain avant le clic
            if self.human_delays:
                time.sleep(random.uniform(0.05, 0.15))
            
            # Clic avec randomisation
            pyautogui.click(x, y, button='left')
            
            # Délai après le clic
            if self.human_delays:
                time.sleep(random.uniform(0.1, 0.3))
            
            self.last_action_time = time.time()
            self.logger.info(f"Clic sur {button_name} à ({x}, {y})")
            
        except Exception as e:
            self.logger.error(f"Erreur clic {button_name}: {e}")
    
    def drag_bet_slider(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]):
        """
        Glisse le slider de mise
        """
        try:
            self.ensure_minimum_delay()
            
            # Randomisation des positions
            start_x = start_pos[0] + random.randint(-3, 3)
            start_y = start_pos[1] + random.randint(-3, 3)
            end_x = end_pos[0] + random.randint(-3, 3)
            end_y = end_pos[1] + random.randint(-3, 3)
            
            # Mouvement vers la position de départ
            self.move_to_coordinates_curved((start_x, start_y))
            
            # Délai avant le drag
            if self.human_delays:
                time.sleep(random.uniform(0.1, 0.2))
            
            # Drag avec vitesse variable
            duration = random.uniform(0.3, 0.8)
            pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)
            
            # Délai après le drag
            if self.human_delays:
                time.sleep(random.uniform(0.2, 0.4))
            
            self.last_action_time = time.time()
            self.logger.info(f"Drag slider de ({start_x}, {start_y}) à ({end_x}, {end_y})")
            
        except Exception as e:
            self.logger.error(f"Erreur drag slider: {e}")
    
    def type_bet_amount(self, amount: int):
        """
        Tape un montant de mise
        """
        try:
            self.ensure_minimum_delay()
            
            # Conversion en string
            amount_str = str(amount)
            
            # Délai avant la frappe
            if self.human_delays:
                time.sleep(random.uniform(0.1, 0.3))
            
            # Frappe avec délais variables entre les caractères
            for char in amount_str:
                pyautogui.press(char)
                time.sleep(random.uniform(0.05, 0.15))
            
            # Délai après la frappe
            if self.human_delays:
                time.sleep(random.uniform(0.2, 0.4))
            
            self.last_action_time = time.time()
            self.logger.info(f"Frappe montant: {amount}")
            
        except Exception as e:
            self.logger.error(f"Erreur frappe montant: {e}")
    
    def emergency_fold(self):
        """
        Fold d'urgence en cas de problème
        """
        try:
            # Recherche rapide du bouton fold
            fold_button_pos = self.find_fold_button()
            if fold_button_pos:
                self.click_button("FOLD_EMERGENCY", fold_button_pos)
                self.logger.warning("Fold d'urgence exécuté")
            else:
                # Fallback: touche F12
                keyboard.press('f12')
                time.sleep(0.1)
                keyboard.release('f12')
                self.logger.warning("Fold d'urgence via F12")
                
        except Exception as e:
            self.logger.error(f"Erreur fold d'urgence: {e}")
    
    def move_to_coordinates_curved(self, target: Tuple[int, int]):
        """
        Mouvement courbe vers les coordonnées cibles
        """
        try:
            # Position actuelle de la souris
            current_pos = pyautogui.position()
            
            # Calcul du chemin courbe
            curve_points = self.calculate_curve_path(current_pos, target)
            
            # Mouvement le long de la courbe
            for point in curve_points:
                pyautogui.moveTo(point[0], point[1], 
                               duration=random.uniform(0.01, 0.03))
            
            # Ajouter à l'historique
            self.movement_history.append({
                'from': current_pos,
                'to': target,
                'time': time.time()
            })
            
            # Limiter l'historique
            if len(self.movement_history) > self.max_history_size:
                self.movement_history.pop(0)
                
        except Exception as e:
            self.logger.error(f"Erreur mouvement courbe: {e}")
    
    def calculate_curve_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> list:
        """
        Calcule un chemin courbe entre deux points
        """
        try:
            # Point de contrôle pour la courbe (décalé aléatoirement)
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            
            # Décalage aléatoire pour éviter les lignes droites
            offset_x = random.randint(-50, 50)
            offset_y = random.randint(-50, 50)
            
            control_point = (mid_x + offset_x, mid_y + offset_y)
            
            # Générer des points le long de la courbe de Bézier
            points = []
            num_points = random.randint(5, 10)
            
            for i in range(num_points + 1):
                t = i / num_points
                x = (1 - t)**2 * start[0] + 2 * (1 - t) * t * control_point[0] + t**2 * end[0]
                y = (1 - t)**2 * start[1] + 2 * (1 - t) * t * control_point[1] + t**2 * end[1]
                points.append((int(x), int(y)))
            
            return points
            
        except Exception as e:
            self.logger.error(f"Erreur calcul chemin courbe: {e}")
            return [start, end]
    
    def find_fold_button(self) -> Optional[Tuple[int, int]]:
        """
        Trouve le bouton fold sur l'écran
        """
        try:
            # Recherche par template matching (simplifié)
            # En production, utiliserait des templates d'images
            
            # Positions typiques du bouton fold
            possible_positions = [
                (400, 600),  # Position typique
                (350, 600),  # Position alternative
                (450, 600),  # Position alternative
            ]
            
            # Retourner la première position valide
            for pos in possible_positions:
                if self.is_valid_screen_position(pos):
                    return pos
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erreur recherche bouton fold: {e}")
            return None
    
    def is_valid_screen_position(self, pos: Tuple[int, int]) -> bool:
        """
        Vérifie si une position est valide sur l'écran
        """
        try:
            screen_width, screen_height = pyautogui.size()
            return 0 <= pos[0] < screen_width and 0 <= pos[1] < screen_height
        except:
            return False
    
    def ensure_minimum_delay(self):
        """
        Assure un délai minimum entre les actions
        """
        current_time = time.time()
        time_since_last = current_time - self.last_action_time
        
        if time_since_last < self.min_delay_between_actions:
            sleep_time = self.min_delay_between_actions - time_since_last
            time.sleep(sleep_time)
    
    def add_human_delay(self):
        """
        Ajoute un délai humain aléatoire
        """
        if self.human_delays:
            delay = random.uniform(self.min_delay_between_actions, self.max_delay_between_actions)
            time.sleep(delay)
    
    def perform_action_sequence(self, actions: list):
        """
        Exécute une séquence d'actions avec délais naturels
        """
        try:
            for action in actions:
                action_type = action.get('type')
                params = action.get('params', {})
                
                if action_type == 'click':
                    self.click_button(params.get('button_name', ''), 
                                   params.get('coordinates', (0, 0)))
                elif action_type == 'drag':
                    self.drag_bet_slider(params.get('start_pos', (0, 0)),
                                       params.get('end_pos', (0, 0)))
                elif action_type == 'type':
                    self.type_bet_amount(params.get('amount', 0))
                elif action_type == 'delay':
                    time.sleep(params.get('duration', 0.5))
                
                # Délai entre les actions
                self.add_human_delay()
                
        except Exception as e:
            self.logger.error(f"Erreur séquence d'actions: {e}")
    
    def set_click_randomization(self, pixels: int):
        """
        Définit la randomisation des clics
        """
        self.click_randomization = max(0, pixels)
    
    def set_move_speed_range(self, min_speed: float, max_speed: float):
        """
        Définit la plage de vitesse des mouvements
        """
        self.move_speed_range = (min_speed, max_speed)
    
    def set_human_delays(self, enabled: bool):
        """
        Active/désactive les délais humains
        """
        self.human_delays = enabled
    
    def get_movement_statistics(self) -> dict:
        """
        Retourne les statistiques des mouvements
        """
        if not self.movement_history:
            return {}
        
        # Calculer les statistiques
        total_movements = len(self.movement_history)
        avg_distance = 0
        avg_duration = 0
        
        for i in range(1, len(self.movement_history)):
            prev = self.movement_history[i-1]
            curr = self.movement_history[i]
            
            # Distance
            dx = curr['to'][0] - prev['to'][0]
            dy = curr['to'][1] - prev['to'][1]
            distance = math.sqrt(dx*dx + dy*dy)
            avg_distance += distance
            
            # Durée
            duration = curr['time'] - prev['time']
            avg_duration += duration
        
        if total_movements > 1:
            avg_distance /= (total_movements - 1)
            avg_duration /= (total_movements - 1)
        
        return {
            'total_movements': total_movements,
            'avg_distance': avg_distance,
            'avg_duration': avg_duration
        }
    
    def detect_patterns(self) -> bool:
        """
        Détecte les patterns répétitifs dans les mouvements
        """
        if len(self.movement_history) < 3:
            return False
        
        # Vérifier les patterns simples
        recent_movements = self.movement_history[-3:]
        
        # Pattern de répétition de positions
        positions = [m['to'] for m in recent_movements]
        if len(set(positions)) == 1:  # Même position répétée
            return True
        
        # Pattern de timing
        timings = [m['time'] for m in recent_movements]
        if len(set(timings)) == 1:  # Même timing
            return True
        
        return False
    
    def add_randomization(self):
        """
        Ajoute de la randomisation supplémentaire
        """
        # Micro-mouvements aléatoires
        current_pos = pyautogui.position()
        offset_x = random.randint(-2, 2)
        offset_y = random.randint(-2, 2)
        
        pyautogui.moveTo(current_pos[0] + offset_x, current_pos[1] + offset_y,
                        duration=random.uniform(0.01, 0.03))
    
    def click_fold(self):
        """Clique sur le bouton Fold"""
        try:
            fold_pos = self.find_fold_button()
            if fold_pos:
                self.click_button("FOLD", fold_pos)
            else:
                self.logger.warning("Bouton fold non trouvé")
        except Exception as e:
            self.logger.error(f"Erreur clic fold: {e}")
    
    def click_call(self):
        """Clique sur le bouton Call"""
        try:
            # Position typique du bouton call
            call_pos = (400, 600)  # À ajuster selon l'interface
            self.click_button("CALL", call_pos)
        except Exception as e:
            self.logger.error(f"Erreur clic call: {e}")
    
    def click_check(self):
        """Clique sur le bouton Check"""
        try:
            # Position typique du bouton check
            check_pos = (300, 600)  # À ajuster selon l'interface
            self.click_button("CHECK", check_pos)
        except Exception as e:
            self.logger.error(f"Erreur clic check: {e}")
    
    def click_raise(self, bet_size: int = 0):
        """Clique sur le bouton Raise et ajuste le montant si nécessaire"""
        try:
            # Position typique du bouton raise
            raise_pos = (500, 600)  # À ajuster selon l'interface
            self.click_button("RAISE", raise_pos)
            
            # Si un montant est spécifié, ajuster le slider
            if bet_size > 0:
                # Position du slider (à ajuster)
                slider_start = (400, 550)
                slider_end = (600, 550)
                self.drag_bet_slider(slider_start, slider_end)
                
                # Ou taper le montant
                self.type_bet_amount(bet_size)
                
        except Exception as e:
            self.logger.error(f"Erreur clic raise: {e}")
    
    def click_all_in(self):
        """Clique sur le bouton All-In"""
        try:
            # Position typique du bouton all-in
            all_in_pos = (600, 600)  # À ajuster selon l'interface
            self.click_button("ALL_IN", all_in_pos)
        except Exception as e:
            self.logger.error(f"Erreur clic all-in: {e}") 

    def click_at_position(self, x: int, y: int, button: str = 'left'):
        """
        Clique à une position spécifique avec mouvement naturel
        """
        try:
            # Vérifier le délai minimum
            self.ensure_minimum_delay()
            
            # Coordonnées avec randomisation
            x += random.randint(-self.click_randomization, self.click_randomization)
            y += random.randint(-self.click_randomization, self.click_randomization)
            
            # Vérifier que la position est valide
            if not self.is_valid_screen_position((x, y)):
                self.logger.warning(f"Position invalide: ({x}, {y})")
                return False
            
            # Mouvement courbe vers la cible
            self.move_to_coordinates_curved((x, y))
            
            # Délai humain avant le clic
            if self.human_delays:
                time.sleep(random.uniform(0.05, 0.15))
            
            # Clic avec randomisation
            pyautogui.click(x, y, button=button)
            
            # Délai après le clic
            if self.human_delays:
                time.sleep(random.uniform(0.1, 0.3))
            
            self.last_action_time = time.time()
            self.logger.info(f"Clic à la position ({x}, {y})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur clic à position ({x}, {y}): {e}")
            return False 