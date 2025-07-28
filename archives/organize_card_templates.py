#!/usr/bin/env python3
"""
🃏 ORGANISATION DES TEMPLATES DE CARTES
========================================

Script pour renommer et organiser tous les templates de cartes
de manière cohérente pour le système de reconnaissance.
"""

import os
import shutil
from pathlib import Path

def organize_card_templates():
    """Organise tous les templates de cartes"""
    print("🃏 ORGANISATION DES TEMPLATES DE CARTES")
    print("=" * 50)
    
    # Mapping des couleurs
    color_mapping = {
        'card_s': '♠',  # Spades
        'card_h': '♥',  # Hearts  
        'card_d': '♦',  # Diamonds
        'card_c': '♣'   # Clubs
    }
    
    # Mapping des rangs
    rank_mapping = {
        'A': 'A', 'a': 'A',
        '2': '2',
        '3': '3',
        '4': '4', 
        '5': '5',
        '6': '6',
        '7': '7',
        '8': '8',
        '9': '9',
        '10': 'T',  # 10 devient T pour cohérence
        'J': 'J', 'j': 'J',
        'Q': 'Q', 'q': 'Q',
        'K': 'K', 'k': 'K'
    }
    
    # Créer le dossier principal pour les templates organisés
    templates_dir = Path("templates/cards/organized")
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    total_renamed = 0
    
    # Parcourir chaque dossier de couleur
    for color_folder, suit_symbol in color_mapping.items():
        color_path = Path(f"templates/cards/{color_folder}")
        
        if not color_path.exists():
            print(f"⚠️  Dossier {color_folder} non trouvé")
            continue
            
        print(f"\n🎨 Traitement de {color_folder} ({suit_symbol})")
        
        # Créer le dossier pour cette couleur
        suit_dir = templates_dir / suit_symbol
        suit_dir.mkdir(exist_ok=True)
        
        # Parcourir tous les fichiers PNG
        for png_file in color_path.glob("*.png"):
            filename = png_file.stem  # Nom sans extension
            
            # Extraire le rang du nom de fichier
            rank = None
            for old_rank, new_rank in rank_mapping.items():
                if filename.upper().startswith(old_rank.upper()):
                    rank = new_rank
                    break
            
            if rank is None:
                print(f"  ⚠️  Rang non reconnu dans {filename}")
                continue
            
            # Nouveau nom de fichier
            new_filename = f"card_{rank}{suit_symbol}.png"
            new_path = suit_dir / new_filename
            
            # Copier le fichier avec le nouveau nom
            try:
                shutil.copy2(png_file, new_path)
                print(f"  ✅ {filename} → {new_filename}")
                total_renamed += 1
            except Exception as e:
                print(f"  ❌ Erreur copie {filename}: {e}")
    
    print(f"\n🎯 RÉSUMÉ")
    print(f"  📁 Templates organisés: {total_renamed}")
    print(f"  📂 Dossier: {templates_dir}")
    
    # Créer un fichier de mapping pour référence
    create_mapping_file(templates_dir, color_mapping, rank_mapping)
    
    return templates_dir

def create_mapping_file(templates_dir, color_mapping, rank_mapping):
    """Crée un fichier de mapping pour référence"""
    mapping_file = templates_dir / "CARD_MAPPING.md"
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write("# 🃏 MAPPING DES CARTES\n\n")
        f.write("## 🎨 Couleurs\n")
        for folder, symbol in color_mapping.items():
            f.write(f"- `{folder}` → `{symbol}`\n")
        
        f.write("\n## 🔢 Rangs\n")
        for old_rank, new_rank in rank_mapping.items():
            f.write(f"- `{old_rank}` → `{new_rank}`\n")
        
        f.write("\n## 📁 Structure des fichiers\n")
        f.write("```\n")
        f.write("templates/cards/organized/\n")
        for symbol in color_mapping.values():
            f.write(f"├── {symbol}/\n")
            f.write(f"│   ├── card_A{symbol}.png\n")
            f.write(f"│   ├── card_2{symbol}.png\n")
            f.write(f"│   ├── ...\n")
            f.write(f"│   └── card_K{symbol}.png\n")
        f.write("└── CARD_MAPPING.md\n")
        f.write("```\n")
    
    print(f"  📄 Mapping créé: {mapping_file}")

if __name__ == "__main__":
    organize_card_templates() 