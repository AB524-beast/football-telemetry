import time
import paho.mqtt.client as mqtt
from config import BROKER_HOST, BROKER_PORT, alert_topic

TOPIC = alert_topic(99)
received = []


def make_client(name):
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"smoke-{name}-{int(time.time())}")
    c.on_message = lambda cl, ud, m: (print(f"[{name}] {m.topic}: {m.payload.decode()} retained={bool(m.retain)}"),
                                      received.append(m.payload))
    c.connect(BROKER_HOST, BROKER_PORT, keepalive=30)
    c.loop_start()
    return c


pub = make_client("publisher")
time.sleep(1)
pub.publish(TOPIC, "TEST: retained hello", qos=1, retain=True).wait_for_publish()
pub.loop_stop(); pub.disconnect()
print("Publisher gone. Message should live on the broker.\n")

time.sleep(1)
sub = make_client("late-subscriber")
sub.subscribe(TOPIC, qos=1)
time.sleep(3)
sub.publish(TOPIC, b"", qos=1, retain=True).wait_for_publish()   # cleanup
sub.loop_stop(); sub.disconnect()
print("\nPASS" if any(received) else "\nFAIL")