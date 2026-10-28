"""Train and test anomalib models on MVTec AD categories.

À ÉCRIRE (semaine 1). Consignes :

Objectif : une commande du type
    python -m inspection.train --model patchcore --category bottle
qui entraîne un modèle sur les images "good" d'une catégorie, l'évalue sur le
split de test (AUROC image et pixel) et, en option, l'exporte en ONNX.

Étapes à coder :
1. Lire les arguments en ligne de commande (module argparse) : modèle, catégorie,
   option d'export.
2. Créer le datamodule anomalib pour MVTec AD, en pointant vers DATA_ROOT
   (voir config.py).
3. Créer le modèle (PatchCore ou PaDiM) à partir de son nom.
4. Créer un Engine anomalib, puis appeler fit, puis test.
5. Si l'option est donnée, exporter le modèle en ONNX.

Indices :
- Commence par lire la doc "Getting started" d'anomalib pour la version installée
  (`pip show anomalib`) : les noms de classes ont changé entre les versions 1.x et 2.x.
- Fais d'abord marcher les étapes 2 à 4 dans un notebook ou un petit script jetable,
  puis range le code ici.
- Garde les imports d'anomalib à l'intérieur des fonctions : la CI n'installe pas
  anomalib, un import en haut du fichier ferait échouer les tests.

Questions à pouvoir expliquer ensuite :
- Pourquoi PatchCore n'a-t-il besoin que d'une seule "époque" d'entraînement ?
- Que contient exactement le split de test de MVTec AD ?
"""

import argparse
import json

from inspection.config import CATEGORIES, DATA_ROOT, MODELS, RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and test anomalib models on MVTec AD categories."
    )
    parser.add_argument(
        "--model", choices=MODELS, required=True, help="Name of the model to train."
    )
    parser.add_argument(
        "--category", choices=CATEGORIES, required=True, help="MVTec AD category to train on."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def train(model_name, category, seed):
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Padim, Patchcore
    from lightning import seed_everything

    seed_everything(seed)

    # 1. Load the dataset
    datamodule = MVTecAD(root=DATA_ROOT, category=category, num_workers=0)

    # 2. Create the model
    if model_name == "patchcore":
        model = Patchcore()
    elif model_name == "padim":
        model = Padim()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # 3. Create the engine
    engine = Engine()

    # 4. Train and evaluate
    engine.fit(model=model, datamodule=datamodule)
    results = engine.test(model=model, datamodule=datamodule)
    return results[0]


def main():
    args = parse_args()
    print(args)
    results = train(args.model, args.category, args.seed)
    results["model"] = args.model
    results["category"] = args.category
    results["seed"] = args.seed
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{args.model}_{args.category}_{args.seed}_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
