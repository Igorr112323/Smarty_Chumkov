from pathlib import Path

LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"
POINTER_MAX_BYTES = 10 * 1024
READY_MIN_BYTES = 100 * 1024


class LfsPointerError(RuntimeError):
    pass


def ensure_readable(path):
    target = Path(path)
    if not target.exists():
        raise LfsPointerError(f"{target}: файл не получен")
    size = target.stat().st_size
    with open(target, "rb") as handle:
        head = handle.read(len(LFS_MAGIC))
    if size < POINTER_MAX_BYTES and head.startswith(LFS_MAGIC):
        raise LfsPointerError(f"{target}: указатель LFS, данные не получены")
    if size < READY_MIN_BYTES:
        raise LfsPointerError(f"{target}: файл не получен, размер {size} байт меньше 100 КБ")
    return size
