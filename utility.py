import time
from datetime import datetime

class AlertLogger:
    def __init__(self, log_file="alert_log.txt"):
        self.log_file = log_file
        self.start_time = time.time()
        self.last_alert_time = 0
        self.cooldown = 2  # seconds between alerts (to prevent spamming)

    def _get_timestamp(self):
        elapsed = time.time() - self.start_time
        return time.strftime('%H:%M:%S', time.gmtime(elapsed))

    def log(self, reason):
        current_time = time.time()
        if current_time - self.last_alert_time < self.cooldown:
            return  # Skip alert if cooldown hasn't passed

        timestamp = self._get_timestamp()
        message = f"[ALERT] {timestamp} - {reason}"

        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

        self.last_alert_time = current_time
if __name__ == "__main__":
    logger = AlertLogger()
    logger.log("This is a test alert.")
