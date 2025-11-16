#!/usr/bin/env python3
import argparse
import csv
import datetime
import json
import logging
import os
import sys
import threading
import time
from typing import Dict, List
from paho.mqtt import client as mqtt
import msgpack

# ---- Label map (edit as needed) ----
LABEL_MAP = {
    "NONE": 0,
    "SLIDE_LEFT": 1,
    "SLIDE_RIGHT": 2,
    "WRIST_TURN_CLOCKWISE": 3,
    "WRIST_TURN_ANTI_CLOCKWISE": 4,
    "SLIDE_UP": 5,
    "SLIDE_DOWN": 6,
    "GRASP": 7,
}

# ---- The exact CSV header you requested (typos preserved on purpose) ----
CSV_COLUMNS = [
    "timestamp",
    # IMU0 (wrist)
    "Imu0_linear_accleration_x","Imu0_linear_accleration_y","Imu0_linear_accleration_z",
    "Imu0_angular_velocity_x","Imu0_angular_velocity_y","Imu0_angular_velocity_z",
    # IMU1 (finger1)
    "Imu1_linear_accleration_x","Imu1_linear_accleration_y","Imu1_linear_accleration_z",
    "Imu1_angular_velocity_x","Imu1_angular_velocity_y","Imu1_angular_velocity_z",
]

# ---- Global counters for rate tracking ----
msg_count = 0
last_print = time.time()

# ---- Global state for timestamp synchronization ----
last_esp32_timestamp = None  # Last received ESP32 timestamp (ms)
last_python_time = None      # Python time when last ESP32 timestamp was received

# ---- Global state for windowed collection ----
collection_state = "IDLE"  # States: "IDLE", "COLLECTING"
collection_buffer = []  # Buffer to store samples during collection window
collection_start_ts = None  # ESP32 timestamp to start collecting from
collection_end_ts = None    # ESP32 timestamp to stop collecting at
COLLECTION_WINDOW_MS = 60000  # milliseconds of IMU data to collect
PRE_BUFFER_MS = 100  # milliseconds to collect before keypress to avoid missing samples

# ---- Thread safety ----
state_lock = threading.Lock()  # Protects collection state and timestamp variables

# ---- Flag for keypress detection ----
enter_pressed = False

def now_stamp_str() -> str:
    # e.g., 20251016_143052_123
    now = datetime.datetime.now()
    ms = now.microsecond // 1000
    return now.strftime(f"%Y%m%d_%H%M%S_{ms:03d}")

def save_buffered_data(out_dir: str, all_samples: List[Dict], label_name: str):
    """Save all buffered samples to a single CSV file"""
    if not all_samples:
        return
    
    os.makedirs(out_dir, exist_ok=True)
    
    ts_name = now_stamp_str()
    csv_path = os.path.join(out_dir, f"{ts_name}.csv")
    
    # Separate IMU0 and IMU1 samples, then pair by closest timestamp
    rows = []
    imu0_samples = []
    imu1_samples = []
    
    # Extract all samples from all buffered payloads
    for sample in all_samples:
        ts = sample.get("ts", 0)
        imu_id = sample.get("imu", 0)
        
        data = {
            "ts": ts,
            "ax": sample.get("ax", ""),
            "ay": sample.get("ay", ""),
            "az": sample.get("az", ""),
            "gx": sample.get("gx", ""),
            "gy": sample.get("gy", ""),
            "gz": sample.get("gz", ""),
        }
        
        if imu_id == 0:
            imu0_samples.append(data)
        elif imu_id == 1:
            imu1_samples.append(data)
    
    # Sort both lists by timestamp
    imu0_samples.sort(key=lambda x: x["ts"])
    imu1_samples.sort(key=lambda x: x["ts"])
    
    # For each IMU0 sample, find the closest IMU1 sample
    for imu0 in imu0_samples:
        row = {c: "" for c in CSV_COLUMNS}
        row["timestamp"] = imu0["ts"]
        
        # Fill IMU0 data
        row["Imu0_linear_accleration_x"] = imu0["ax"]
        row["Imu0_linear_accleration_y"] = imu0["ay"]
        row["Imu0_linear_accleration_z"] = imu0["az"]
        row["Imu0_angular_velocity_x"] = imu0["gx"]
        row["Imu0_angular_velocity_y"] = imu0["gy"]
        row["Imu0_angular_velocity_z"] = imu0["gz"]
        
        # Find closest IMU1 sample
        if imu1_samples:
            closest_imu1 = None
            min_diff = float('inf')
            
            # Search for the closest timestamp
            for imu1 in imu1_samples:
                diff = abs(imu1["ts"] - imu0["ts"])
                if diff < min_diff:
                    min_diff = diff
                    closest_imu1 = imu1
                elif diff > min_diff:
                    # Since sorted, if diff starts increasing, we've passed the closest
                    break
            
            # Fill IMU1 data if found
            if closest_imu1:
                row["Imu1_linear_accleration_x"] = closest_imu1["ax"]
                row["Imu1_linear_accleration_y"] = closest_imu1["ay"]
                row["Imu1_linear_accleration_z"] = closest_imu1["az"]
                row["Imu1_angular_velocity_x"] = closest_imu1["gx"]
                row["Imu1_angular_velocity_y"] = closest_imu1["gy"]
                row["Imu1_angular_velocity_z"] = closest_imu1["gz"]
        
        rows.append(row)
    
    # Write CSV (header + multiple rows)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    
    logging.info(f"[WRITE] {os.path.basename(csv_path)}  {len(rows)} rows  label={label_name}({LABEL_MAP.get(label_name.upper(), LABEL_MAP['NONE'])})")
    
    # Update labels.json
    label_int = LABEL_MAP.get(label_name.upper(), LABEL_MAP["NONE"])
    labels_path = os.path.join(out_dir, "labels.json")
    labels: Dict[str, int] = {}
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r") as f:
                labels = json.load(f)
        except Exception:
            labels = {}
    
    labels[ts_name] = label_int
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2, sort_keys=True)


def keyboard_listener():
    """Background thread that listens for Enter keypresses"""
    global enter_pressed, collection_state, collection_start_ts, collection_end_ts
    global last_esp32_timestamp, last_python_time
    
    print("\n=== Ready for data collection ===")
    print("Press ENTER to start collecting 1 second of gesture data...")
    
    while True:
        try:
            input()  # Wait for Enter keypress
            
            with state_lock:
                # Check if we have received any ESP32 data yet
                if last_esp32_timestamp is None or last_python_time is None:
                    print("[WARN] No ESP32 data received yet. Waiting for MQTT messages...")
                    continue
                
                # Check if already collecting
                if collection_state == "COLLECTING":
                    print("[WARN] Already collecting data. Please wait...")
                    continue
                
                # Calculate estimated current ESP32 timestamp
                python_elapsed_ms = (time.time() - last_python_time) * 1000
                estimated_ts = last_esp32_timestamp + python_elapsed_ms
                
                # Set collection window (start PRE_BUFFER_MS before to catch any missed samples)
                collection_start_ts = estimated_ts - PRE_BUFFER_MS
                collection_end_ts = estimated_ts + COLLECTION_WINDOW_MS
                
                # Transition to COLLECTING state
                collection_state = "COLLECTING"
                
                logging.info(f"[COLLECT] Started! Window: {collection_start_ts:.0f} to {collection_end_ts:.0f} ms (ESP32 time, with {PRE_BUFFER_MS}ms pre-buffer)")
            
        except EOFError:
            # Handle Ctrl+D or piped input
            break
        except Exception as e:
            logging.error(f"[ERROR] Keyboard listener error: {e}")

def decode_payload(payload: bytes):
    """Try to decode as MsgPack first, then JSON, else return raw bytes"""
    # First try MsgPack (what the ESP32 is sending)
    try:
        return "msgpack", msgpack.unpackb(payload, raw=False)
    except Exception:
        pass
    # Then try JSON
    try:
        return "json", json.loads(payload.decode("utf-8"))
    except Exception:
        pass
    return "bytes", payload

def on_connect(client, userdata, flags, rc):
    """Callback when MQTT connection is established"""
    if rc == 0:
        logging.info(f"[connect] OK (rc={rc}) — subscribing to '{userdata['topic']}'")
        client.subscribe(userdata["topic"], qos=userdata["qos"])
    else:
        logging.error(f"[connect] FAILED rc={rc}")

def on_message(client, userdata, msg):
    """Callback when a message is received on the subscribed topic"""
    global collection_buffer, collection_state, collection_start_ts, collection_end_ts
    global last_esp32_timestamp, last_python_time

    # Decode the message
    kind, obj = decode_payload(msg.payload)
    
    if kind not in ("msgpack", "json"):
        logging.warning(f"[WARN] Non-decodable payload (type={kind}), skipping")
        return
    
    try:
        # Extract samples - now expecting single sample per message or batch format
        samples = []
        if "samples" in obj and isinstance(obj["samples"], list):
            # Still handle batch format if it comes
            samples = obj["samples"]
        elif "ts" in obj and "imu" in obj:
            # Single sample format
            samples = [obj]
        else:
            logging.warning("[WARN] Unrecognized message format, skipping")
            return
        
        # Process each sample
        for sample in samples:
            sample_ts = sample.get("ts", None)
            if sample_ts is None:
                continue
            
            with state_lock:
                # Always update timestamp tracking for synchronization
                last_esp32_timestamp = sample_ts
                last_python_time = time.time()
                
                # Get current state (inside lock)
                current_state = collection_state
            
            # Handle based on current state
            if current_state == "IDLE":
                # Not collecting, just update timestamps (already done above)
                pass
                
            elif current_state == "COLLECTING":
                buffer_to_save = None
                should_save = False
                
                # Check if sample is within collection window
                with state_lock:
                    if collection_start_ts <= sample_ts <= collection_end_ts:
                        # Buffer this sample
                        collection_buffer.append(sample)
                    
                    # Check if we've passed the end of collection window
                    if sample_ts >= collection_end_ts:
                        # Collection window ended, save data
                        num_imu0 = sum(1 for s in collection_buffer if s.get("imu") == 0)
                        num_imu1 = sum(1 for s in collection_buffer if s.get("imu") == 1)
                        logging.info(f"[COLLECT] Window complete: {len(collection_buffer)} samples (IMU0:{num_imu0}, IMU1:{num_imu1})")
                        
                        # Copy buffer for saving (release lock during I/O)
                        buffer_to_save = collection_buffer.copy()
                        should_save = True
                        
                        # Reset for next collection window (inside lock)
                        collection_buffer = []
                        collection_state = "IDLE"
                        collection_start_ts = None
                        collection_end_ts = None
                
                # Save outside the lock to avoid blocking MQTT thread during I/O
                if should_save and buffer_to_save:
                    save_buffered_data(userdata["out_dir"], buffer_to_save, userdata["label"])
                    
                    print("\n=== Ready for next gesture ===")
                    print("Press ENTER to collect data...")

                    
    except Exception as e:
        logging.error(f"[ERROR] Failed to process message: {e}")

def main():
    ap = argparse.ArgumentParser(
        description="Subscribe to MQTT (no TLS), decode MsgPack/JSON, and save to CSV + labels.json"
    )
    ap.add_argument("-H", "--host", default="laptop.local", help="Broker host (default: laptop.local)")
    ap.add_argument("-p", "--port", type=int, default=1883, help="Broker port (default: 1883)")
    ap.add_argument("-t", "--topic", default="b14/esp32/esp32-001/telemetry", help="Topic to subscribe")
    ap.add_argument("--qos", type=int, default=0, choices=[0, 1, 2], help="QoS (default: 0)")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent (default: 2)")
    ap.add_argument("--out", default="data", help="Output directory for CSV files")
    ap.add_argument("--label", default="NONE", help=f"Label for data: {', '.join(LABEL_MAP.keys())}")
    args = ap.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Validate label
    if args.label.upper() not in LABEL_MAP:
        logging.warning(f"Unknown label '{args.label}', defaulting to NONE (0)")
        args.label = "NONE"

    userdata = {
        "topic": args.topic,
        "qos": args.qos,
        "indent": args.indent,
        "out_dir": args.out,
        "label": args.label,
    }
    
    client = mqtt.Client(userdata=userdata)

    client.on_connect = on_connect
    client.on_message = on_message

    logging.info(f"[connect] host={args.host} port={args.port} topic='{args.topic}' qos={args.qos}")
    logging.info(f"[output] dir='{args.out}' label='{args.label}' ({LABEL_MAP.get(args.label.upper(), 0)})")
    logging.info(f"[timing] Collection: {COLLECTION_WINDOW_MS}ms of IMU data (manual trigger)")
    
    # Start keyboard listener thread
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    try:
        client.connect(args.host, args.port, keepalive=30)
    except Exception as e:
        logging.error(f"[connect] error: {e}")
        sys.exit(2)

    try:
        client.loop_forever(retry_first_connection=True)
    except KeyboardInterrupt:
        logging.info("[exit] bye")

if __name__ == "__main__":
    main()

