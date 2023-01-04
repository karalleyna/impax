from impax.utils.base.log import Log


class SimpleLog(Log):
    """A log that just prints with a level indicator."""

    def __init__(self):
        super(SimpleLog, self).__init__()
        self.visible_levels = self.levels

    def log(self, msg, level="info"):
        if level.lower() not in self.levels:
            raise ValueError(f"Invalid logging level: {level}")
        if level.lower() not in self.visible_levels:
            return  # Too low level to display
        print(f"{level.upper()}: {msg}")

    def verbose(self, msg):
        self.log(msg, level="verbose")

    def info(self, msg):
        self.log(msg, level="info")

    def warning(self, msg):
        self.log(msg, level="warning")

    def error(self, msg):
        self.log(msg, level="error")

    def set_level(self, level):
        index = self.level_index(level)
        self.visible_levels = self.levels[index:]
        self.verbose(f"Logging level changed to {level}")
