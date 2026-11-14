BROKER_HOST = "broker.emqx.io"
BROKER_PORT = 1883

TOPIC_ROOT = "ammyb-11"   # CHANGE THIS to something unique
TEAM = "teamA"
NUM_PLAYERS = 11
PUBLISH_INTERVAL_S = 2

WINDOW_SIZE = 20            # samples per player (~40 s at 2 s/sample)
MODEL_PATH = "fatigue_model.joblib"
PROFILES_PATH = "data/profiles.json"
ALERT_THRESHOLD = 0.80      # P(fatigued) above this counts as "high"
SUSTAINED_PREDICTIONS = 3   # consecutive high predictions before alerting
ALERT_COOLDOWN_S = 60


def biometrics_topic(pid: int) -> str:
    return f"{TOPIC_ROOT}/{TEAM}/player{pid:02d}/biometrics"


def fatigue_topic(pid: int) -> str:
    return f"{TOPIC_ROOT}/{TEAM}/player{pid:02d}/fatigue"


def alert_topic(pid: int) -> str:
    return f"{TOPIC_ROOT}/{TEAM}/alerts/injury/player{pid:02d}"


BIOMETRICS_WILDCARD = f"{TOPIC_ROOT}/{TEAM}/+/biometrics"
FATIGUE_WILDCARD = f"{TOPIC_ROOT}/{TEAM}/+/fatigue"
ALERT_WILDCARD = f"{TOPIC_ROOT}/{TEAM}/alerts/injury/+"