from impax.utils.base.file_system import FileSystem
import glob
import os
import shutil


class StandardFileSystem(FileSystem):
    """A FileSystem that uses the standard os and built-in modules."""

    def mkdir(self, path, exist_ok=False):
        try:
            os.mkdir(path)
        except FileExistsError as e:
            if exist_ok:
                return
            raise FileExistsError("Passing through mkdir() error.") from e

    def makedirs(self, path, exist_ok=False):
        return os.makedirs(path, exist_ok=exist_ok)

    def open(self, *args):
        return open(*args)

    def glob(self, *args):
        return glob.glob(*args)

    def exists(self, *args):
        return os.path.exists(*args)

    def cp(self, *args):
        return shutil.copyfile(*args)

    def rm(self, *args):
        return os.remove(*args)
