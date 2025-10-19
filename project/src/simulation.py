from logging import Logger


class Simulation:
    timestep: int

    def __init__(self, logger: Logger):
        self.logger = logger
        self.timestep = 0

    def run(self):
        self.logger.info("simulation started")
        self.logger.info("simulation finished")
