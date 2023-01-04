import abc


class FileSystem(abc.ABC):
    """An abstract base class representing an interface to a filesystem."""

    @abc.abstractmethod
    def open(self, filename, mode):
        pass

    @abc.abstractmethod
    def mkdir(self, path, exist_ok=False):
        """Makes a directory.
        Args:
          path: String. The path to the directory to make.
          exist_ok: Boolean. If True, errors that the file already exists are
            suppressed. Otherwise, the function raises an exception if the directory
            already exists.
        """
        pass

    @abc.abstractmethod
    def makedirs(self, path, exist_ok=False):
        """Makes a directory tree recursively.
        Args:
          path: String. The path to the directory to make.
          exist_ok: Boolean. If True, errors that the file already exists are
            suppressed. Otherwise, the function raises an exception if the directory
            already exists.
        """
        pass

    @abc.abstractmethod
    def glob(self, path):
        pass

    @abc.abstractmethod
    def exists(self, path):
        pass

    @abc.abstractmethod
    def cp(self, from_path, to_path):
        """Copies a regular file (not a directory) to a new location.
        If a file already exists at the destination, or the source does not exist
        or is a directory, then behavior is unspecified.
        Args:
          from_path: String. The path to the source file.
          to_path: String. The path to the destination file.
        """
        # TODO(kgenova) This behavior should be better specified.
        pass

    def rm(self, path):
        """Removes a regular file (not a directory).
        If the file does not exist, permissions are insufficient, or the path
        points to a directory, then behavior is unspecified.
        Args:
          path: String. The path to the file to be removed.
        """
        pass
