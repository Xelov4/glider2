#!/usr/bin/env python3
"""
🔧 CORRECTION DES NOMS DE FICHIERS TEMPLATES
============================================

Script pour corriger les noms de fichiers des templates en utilisant des caractères ASCII.
"""

import os
import shutil
from pathlib import Path

def fix_template_names():
    """Corrige les noms de fichiers des templates"""
    print("🔧 CORRECTION DES NOMS DE FICHIERS TEMPLATES")
    print("=" * 50)
    
    # Mapping des couleurs avec caractères ASCII
    color_mapping = {
        '♠': 'spades',    # Spades
        '♥': 'hearts',    # Hearts  
        '♦': 'diamonds',  # Diamonds
        '♣': 'clubs'      # Clubs
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
    
    # Créer le dossier principal pour les templates corrigés
    templates_dir = Path("templates/cards/fixed")
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    total_renamed = 0
    
    # Parcourir chaque dossier de couleur dans organized
    organized_dir = Path("templates/cards/organized")
    
    for color_symbol, ascii_name in color_mapping.items():
        color_path = organized_dir / color_symbol
        
        if not color_path.exists():
            print(f"⚠️  Dossier {color_symbol} non trouvé")
            continue
            
        print(f"\n🎨 Traitement de {color_symbol} ({ascii_name})")
        
        # Créer le dossier pour cette couleur
        suit_dir = templates_dir / ascii_name
        suit_dir.mkdir(exist_ok=True)
        
        # Parcourir tous les fichiers PNG
        for png_file in color_path.glob("*.png"):
            filename = png_file.stem  # Nom sans extension
            
            # Extraire le rang du nom de fichier (format: card_A♠)
            if filename.startswith("card_"):
                # Enlever "card_" et le symbole de couleur
                rank_part = filename[5:-1]  # Enlever "card_" et le dernier caractère (symbole)
                
                # Mapping du rang
                rank = rank_mapping.get(rank_part, rank_part)
                
                # Nouveau nom de fichier avec caractères ASCII
                new_filename = f"card_{rank}_{ascii_name}.png"
                new_path = suit_dir / new_filename
                
                # Copier le fichier avec le nouveau nom
                try:
                    shutil.copy2(png_file, new_path)
                    print(f"  ✅ {filename} → {new_filename}")
                    total_renamed += 1
                except Exception as e:
                    print(f"  ❌ Erreur copie {filename}: {e}")
    
    print(f"\n🎯 RÉSUMÉ")
    print(f"  📁 Templates corrigés: {total_renamed}")
    print(f"  📂 Dossier: {templates_dir}")
    
    # Créer un fichier de mapping pour référence
    create_mapping_file(templates_dir, color_mapping, rank_mapping)
    
    return templates_dir

def create_mapping_file(templates_dir, color_mapping, rank_mapping):
    """Crée un fichier de mapping pour référence"""
    mapping_file = templates_dir / "CARD_MAPPING.md"
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write("# 🃏 MAPPING DES CARTES (ASCII)\n\n")
        f.write("## 🎨 Couleurs\n")
        for symbol, ascii_name in color_mapping.items():
            f.write(f"- `{symbol}` → `{ascii_name}`\n")
        
        f.write("\n## 🔢 Rangs\n")
        for old_rank, new_rank in rank_mapping.items():
            f.write(f"- `{old_rank}` → `{new_rank}`\n")
        
        f.write("\n## 📁 Structure des fichiers\n")
        f.write("```\n")
        f.write("templates/cards/fixed/\n")
        for ascii_name in color_mapping.values():
            f.write(f"├── {ascii_name}/\n")
            f.write(f"│   ├── card_A_{ascii_name}.png\n")
            f.write(f"│   ├── card_2_{ascii_name}.png\n")
            f.write(f"│   ├── ...\n")
            f.write(f"│   └── card_K_{ascii_name}.png\n")
        f.write("└── CARD_MAPPING.md\n")
        f.write("```\n")
    
    print(f"  📄 Mapping créé: {mapping_file}")

if __name__ == "__main__":
    fix_template_names() 