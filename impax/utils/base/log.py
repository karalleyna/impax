import abc


class Log(abc.ABC):
    """An abstract class representing a log for messages."""

    @abc.abstractmethod
    def log(self, msg, level="info"):
        """Logs a message to the underlying log."""
        pass

    @property
    def levels(self):
        return ["verbose", "info", "warning", "error"]

    def level_index(self, level):
        level = level.lower()
        if level not in self.levels:
            raise ValueError(f"Unrecognized logging level: {level}")
        i = 0
        for i in range(len(self.levels)):
            if self.levels[i] == level:
                return i
        assert False  # Should be unreachable
