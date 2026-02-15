import os
import json
import random
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# ================= Configuration =================
# Default split ratio for Train/Val/Test
SPLIT_RATIO = [0.8, 0.1, 0.1]
RANDOM_SEED = 42


# =============================================

class GraphBuilder:
    """
    Core logic for converting traffic flows into graph structures.
    Includes dynamic thresholding and statistical feature extraction.
    """

    @staticmethod
    def calculate_dynamic_threshold(flows: List[Dict], min_thresh: float = 0.5, max_thresh: float = 5.0) -> float:
        """
        Calculates the adaptive burst threshold based on Inter-Arrival Time (IAT) outliers.
        Logic: Use IQR (Inter-Quartile Range) to identify silence intervals significantly longer than normal.
        """
        if len(flows) < 2:
            return 1.0

        # 1. Extract timestamps
        timestamps = sorted([f.get('start_timestamp', 0) for f in flows])

        # 2. Calculate IAT
        iats = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

        # 3. Filter out micro-intervals (e.g., concurrent flows < 10ms)
        valid_iats = [t for t in iats if t > 0.01]

        if not valid_iats:
            return 1.0

        # 4. Calculate IQR
        q1, q3 = np.percentile(valid_iats, [25, 75])
        iqr = q3 - q1

        # 5. Upper Fence for outliers
        dynamic_threshold = q3 + 1.5 * iqr

        # 6. Clamping
        return max(min_thresh, min(dynamic_threshold, max_thresh))

    @staticmethod
    def extract_node_features(flow: Dict, session_start_time: float, max_seq_len: int = 50) -> Dict:
        """
        Extracts numerical features for a single node (flow).
        Returns structured data, not a string.
        """
        # 1. Sequence Features (Truncated/Padded handled in dataset or here)
        # Here we just store the raw list, truncation happens at tensor conversion
        pkts = flow.get('packet_length', [])
        seq = pkts[:max_seq_len] if pkts else []

        # 2. Statistical Features
        start_ts = flow.get('start_timestamp', 0)
        end_ts = flow.get('end_timestamp', 0)
        deltas = flow.get('arrive_time_delta', [])

        # Relative Start Time (Crucial for anti-drift)
        rel_time = max(0.0, start_ts - session_start_time)

        # Duration
        duration = max(0.0, end_ts - start_ts)

        # Bytes Up/Down Split (Positive=Up, Negative=Down in typical PCAP parsers)
        bytes_up = sum(p for p in pkts if p > 0)
        bytes_down = sum(abs(p) for p in pkts if p < 0)

        # Packet Count
        cnt = len(pkts)

        # Mean IAT
        mean_iat = sum(deltas) / len(deltas) if deltas else 0.0

        return {
            "t": rel_time,
            # Order: [Duration, BytesUp, BytesDown, Count, MeanIAT]
            "stats": [duration, bytes_up, bytes_down, cnt, mean_iat],
            "seq": seq
        }

    @staticmethod
    def build_graph(flows: List[Dict], gamma: float) -> Dict:
        """
        Constructs the graph topology based on the burst threshold (gamma).
        Returns a dictionary containing nodes and adjacency list.
        """
        if not flows:
            return None

        # 1. Sort by time
        sorted_flows = sorted(flows, key=lambda x: x.get('start_timestamp', 0))
        session_start_time = sorted_flows[0].get('start_timestamp', 0)

        # 2. Generate Nodes
        nodes_data = []
        for idx, flow in enumerate(sorted_flows):
            flow['_idx'] = idx  # Temporary ID for edge construction
            nodes_data.append(GraphBuilder.extract_node_features(flow, session_start_time))

        # 3. Burst Segmentation (Window-Based)
        bursts = []
        current_burst = []
        if sorted_flows:
            current_burst.append(sorted_flows[0])
            burst_start = sorted_flows[0].get('start_timestamp', 0)

            for i in range(1, len(sorted_flows)):
                flow = sorted_flows[i]
                curr_time = flow.get('start_timestamp', 0)

                if (curr_time - burst_start) <= gamma:
                    current_burst.append(flow)
                else:
                    bursts.append(current_burst)
                    current_burst = [flow]
                    burst_start = curr_time
            if current_burst:
                bursts.append(current_burst)

        # 4. Generate Edges
        # Format: [Source, Target, Type]
        # Types: 0=Burst(Intra), 1=Seq(Inter), 2=Anchor(Inter)
        edges_data = []

        for b_idx, burst in enumerate(bursts):
            # Type 0: Intra-Burst (Time consistency)
            if len(burst) > 1:
                for k in range(len(burst) - 1):
                    u = burst[k]['_idx']
                    v = burst[k + 1]['_idx']
                    edges_data.append([u, v, 0])

            # Inter-Burst Connections
            if b_idx > 0:
                prev_burst = bursts[b_idx - 1]
                curr_burst = burst

                u_last = prev_burst[-1]['_idx']
                v_first = curr_burst[0]['_idx']

                # Type 1: Sequential (Last -> First)
                edges_data.append([u_last, v_first, 1])

                # Type 2: Anchor (Last -> Last) - Optional but good for stability
                v_last = curr_burst[-1]['_idx']
                if v_first != v_last or len(curr_burst) == 1:
                    edges_data.append([u_last, v_last, 2])

        return {
            "nodes": nodes_data,
            "edges": edges_data
        }


def process_directory(input_dir: str, enable_dynamic: bool) -> List[Dict]:
    """
    Traverses the directory and processes JSON flow files.
    """
    dataset_samples = []
    root_path = Path(input_dir)

    if not root_path.exists():
        print(f"[!] Input directory not found: {input_dir}")
        return []

    print(f"[*] Scanning {input_dir}...")

    # Structure: Root -> App -> Version -> JSONs
    # Adjust depth based on your actual dataset structure
    for app_dir in root_path.iterdir():
        if not app_dir.is_dir(): continue
        app_name = app_dir.name

        # Traverse recursively to find all .json files
        json_files = list(app_dir.rglob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    flows = json.load(f)

                if not flows: continue

                # Calculate Threshold
                if enable_dynamic:
                    gamma = GraphBuilder.calculate_dynamic_threshold(flows)
                else:
                    gamma = 1.0

                # Build Graph
                graph_obj = GraphBuilder.build_graph(flows, gamma)
                if not graph_obj or not graph_obj['nodes']:
                    continue

                # Create Sample
                sample = {
                    "label": app_name,
                    "graph": graph_obj,
                    "meta": {
                        "filename": json_file.name,
                        "gamma": gamma
                    }
                }
                dataset_samples.append(sample)

            except Exception as e:
                print(f"[!] Error processing {json_file}: {e}")

    return dataset_samples


def save_jsonl(data: List[Dict], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    print(f"[+] Saved {len(data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Traffic Graph Preprocessor")
    parser.add_argument("--input", type=str, required=True, help="Path to raw flow JSONs")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--dynamic", action="store_true", default=True, help="Enable dynamic thresholding")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    random.seed(RANDOM_SEED)

    # Process Data
    all_data = process_directory(args.input, args.dynamic)

    if not all_data:
        print("[!] No data processed.")
        return

    # Shuffle and Split (Chronological split is better for drift, but random for I.I.D)
    # Assuming this script handles the I.I.D part.
    # For Drift/OOD, you should run this script separately on the OOD folder.
    random.shuffle(all_data)

    total = len(all_data)
    train_end = int(total * SPLIT_RATIO[0])
    val_end = int(total * (SPLIT_RATIO[0] + SPLIT_RATIO[1]))

    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]

    save_jsonl(train_data, os.path.join(args.output, "train.jsonl"))
    save_jsonl(val_data, os.path.join(args.output, "valid.jsonl"))
    save_jsonl(test_data, os.path.join(args.output, "test.jsonl"))

    print("[*] Preprocessing Complete.")


if __name__ == "__main__":
    main()