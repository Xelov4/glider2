"""
Script pour générer la liste complète des 52 cartes à créer
"""

import os

def create_complete_cards_list():
    """Génère la liste complète des 52 cartes de poker"""
    
    # Rangs et couleurs
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['spades', 'hearts', 'diamonds', 'clubs']
    suit_symbols = ['♠', '♥', '♦', '♣']
    
    print("=== LISTE COMPLÈTE DES 52 CARTES À CRÉER ===\n")
    
    # Créer le dossier complete s'il n'existe pas
    complete_dir = "cards/complete"
    if not os.path.exists(complete_dir):
        os.makedirs(complete_dir)
        print(f"✅ Dossier créé: {complete_dir}")
    
    # Générer la liste complète
    all_cards = []
    
    for suit_idx, suit in enumerate(suits):
        print(f"\n📋 {suit.upper()} ({suit_symbols[suit_idx]}):")
        print("-" * 40)
        
        for rank in ranks:
            filename = f"{rank}{suit}.png"
            card_name = f"{rank}{suit_symbols[suit_idx]}"
            filepath = f"{complete_dir}/{filename}"
            
            # Vérifier si le fichier existe déjà
            if os.path.exists(filepath):
                status = "✅ EXISTE"
            else:
                status = "❌ MANQUANT"
                all_cards.append((filename, card_name))
            
            print(f"  {status} {filename} - {card_name}")
    
    # Résumé
    print(f"\n📊 RÉSUMÉ:")
    print(f"  - Total: 52 cartes")
    print(f"  - Manquantes: {len(all_cards)}")
    print(f"  - Présentes: {52 - len(all_cards)}")
    
    if all_cards:
        print(f"\n🎯 CARTES À CRÉER EN PRIORITÉ:")
        for filename, card_name in all_cards:
            print(f"  - {filename} ({card_name})")
    
    # Créer un fichier de liste
    with open("cards/complete/cards_to_create.txt", "w", encoding="utf-8") as f:
        f.write("LISTE DES CARTES À CRÉER\n")
        f.write("=" * 30 + "\n\n")
        
        for filename, card_name in all_cards:
            f.write(f"{filename} - {card_name}\n")
    
    print(f"\n📄 Liste sauvegardée: cards/complete/cards_to_create.txt")

if __name__ == "__main__":
    create_complete_cards_list() 