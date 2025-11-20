import os


def resolve_image_path(image_path: str | None, root_dir: str | None = None) -> tuple[str | None, str | None]:
    """
    Resolve image_path against root_dir when appropriate.

    Returns a tuple (resolved_path, inferred_root_dir).
    - If image_path is absolute: resolved_path=image_path and inferred_root_dir=os.path.dirname(image_path)
    - If image_path is relative and root_dir provided: join them, return (joined, root_dir)
    - If image_path is relative and no root_dir: return (image_path, None)
    - If image_path is None: return (None, None)
    """
    if image_path is None:
        return None, None
    if os.path.isabs(image_path):
        return os.path.normpath(image_path), os.path.dirname(os.path.normpath(image_path))
    if root_dir:
        return os.path.normpath(os.path.join(root_dir, image_path)), root_dir
    return os.path.normpath(image_path), None

def is_local_path(path: str) -> bool:
    # crude check — treat s3://, gs://, http(s) as non-local
    return not (path.startswith("s3://") or path.startswith("gs://") or path.startswith("http://") or path.startswith("https://"))
