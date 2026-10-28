from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

from inspection.config import DATA_ROOT


def main():
    # 1. Les données
    datamodule = MVTecAD(root=DATA_ROOT, category="screw", num_workers=0)

    # 2. Le modèle
    model = Patchcore()

    # 3. Le chef d'orchestre
    engine = Engine()

    # 4. Construire la mémoire, puis évaluer
    engine.fit(model=model, datamodule=datamodule)
    engine.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
