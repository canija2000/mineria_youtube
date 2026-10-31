"""Sube el servidor + web a un Space de Hugging Face (SDK Docker).

Crea el Space si no existe y sube solo lo que usa el Dockerfile, más `deploy/space_README.md`
como README (ahí está la configuración del Space: sdk, puerto). Con `--datos` sube además el
paquete de `analysis/paquete.py` a un Dataset privado, para usarlo como DATA_URL.

Antes: `hf auth login` (un token con permiso de escritura).
Los secretos (YOUTUBE_API_KEY, HF_TOKEN, DATA_URL) se configuran en la página del Space,
Settings → Variables and secrets; este script no los sube.

Uso:
    python3 deploy/subir_space.py usuario/mapa-youtube
    python3 deploy/subir_space.py usuario/mapa-youtube --datos mineria_youtube_datos.zip
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
ARCHIVOS = ["Dockerfile", ".dockerignore", "requirements-server.txt", "remap.py",
            "recolect/*", "server/*", "web/*"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", help="usuario/nombre del Space")
    parser.add_argument("--publico", action="store_true", help="crea el Space público (por defecto, privado)")
    parser.add_argument("--datos", type=Path, help="zip de datos a subir a un Dataset privado <usuario>/<nombre>-datos")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=not args.publico, exist_ok=True)
    api.upload_folder(folder_path=ROOT, repo_id=args.space, repo_type="space",
                      allow_patterns=ARCHIVOS, ignore_patterns=["**/__pycache__/*", "*.pyc"],
                      commit_message="código del servidor y la web")
    api.upload_file(path_or_fileobj=ROOT / "deploy" / "space_README.md", path_in_repo="README.md",
                    repo_id=args.space, repo_type="space", commit_message="README del Space")
    print(f"Space: https://huggingface.co/spaces/{args.space}")

    if args.datos:
        dataset = f"{args.space}-datos"
        api.create_repo(dataset, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=args.datos, path_in_repo=args.datos.name,
                        repo_id=dataset, repo_type="dataset", commit_message="paquete de datos")
        print(f"Datos: DATA_URL=https://huggingface.co/datasets/{dataset}/resolve/main/{args.datos.name}")


if __name__ == "__main__":
    main()
