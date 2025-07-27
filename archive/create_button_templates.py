"""
Script pour créer les templates de boutons manquants
"""

import cv2
import numpy as np
import os

def create_button_template(button_name: str, width: int = 120, height: int = 40):
    """Crée un template de bouton avec le texte"""
    # Créer une image de base
    img = np.ones((height, width, 3), dtype=np.uint8) * 50  # Fond gris foncé
    
    # Ajouter une bordure
    cv2.rectangle(img, (0, 0), (width-1, height-1), (100, 100, 100), 2)
    
    # Ajouter le texte
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)  # Texte blanc
    
    # Calculer la position du texte
    text_size = cv2.getTextSize(button_name.upper(), font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    # Dessiner le texte
    cv2.putText(img, button_name.upper(), (text_x, text_y), font, font_scale, color, thickness)
    
    return img

def main():
    """Crée tous les templates de boutons manquants"""
    # Créer le dossier templates/buttons s'il n'existe pas
    os.makedirs("templates/buttons", exist_ok=True)
    
    # Liste des boutons à créer
    buttons = [
        'fold_button',
        'call_button', 
        'raise_button',
        'check_button',
        'all_in_button',
        'bet_button'
    ]
    
    print("Création des templates de boutons...")
    
    for button_name in buttons:
        # Créer le template
        template = create_button_template(button_name)
        
        # Sauvegarder
        filename = f"templates/buttons/{button_name}.png"
        cv2.imwrite(filename, template)
        print(f"✅ Template créé: {filename}")
    
    print("\n🎯 Tous les templates de boutons ont été créés !")
    print("Vous pouvez maintenant utiliser l'outil de calibrage pour mapper les boutons.")

if __name__ == "__main__":
    main() 