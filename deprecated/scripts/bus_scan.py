"""Ping the Feetech serial bus directly to find any responding motor.

Run when lerobot reports "Full found motor list: {}" — this scans IDs 1..20
at every common baud rate and prints whatever answers. If something replies,
the cabling and power are fine and only the IDs/baud are wrong (re-run
lerobot-setup-motors). If nothing replies, the issue is power or wiring.

Usage: python scripts/bus_scan.py [/dev/ttyACM0]
"""
import os
import sys
import time
from glob import glob

from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS

PORT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FOLLOWER_PORT", "/dev/ttyACM0")
BAUDS = [1_000_000, 500_000, 250_000, 128_000, 115_200, 57_600, 38_400, 19_200, 14_400, 9_600, 4_800]
MAX_ID = 20


def print_available_ports():
    ports = sorted(glob("/dev/tty.usb*") + glob("/dev/cu.usb*"))
    print(f"USB serial ports visible now: {ports or '(none)'}")


def main() -> int:
    print_available_ports()
    print(f"Scanning {PORT}")

    ph = PortHandler(PORT)
    if not ph.openPort():
        print(f"Could not open {PORT}")
        return 1
    pkt = PacketHandler(0)

    found = []
    for baud in BAUDS:
        if not ph.setBaudRate(baud):
            continue
        print(f"\n--- {baud} baud ---")
        any_hit = False

        # This mirrors LeRobot's protocol-0 bus discovery more closely than
        # sequential ping. A missing response here and below means the serial
        # adapter is alive but no powered servo is answering on the TTL bus.
        if hasattr(pkt, "broadcastPing"):
            ids_status, comm_result = pkt.broadcastPing(ph)
            if comm_result == COMM_SUCCESS and ids_status:
                for sid in sorted(ids_status):
                    model_no, model_comm, _ = pkt.ping(ph, sid)
                    if model_comm == COMM_SUCCESS:
                        print(f"  broadcast id={sid:3d}  model={model_no}")
                        found.append((baud, sid, model_no))
                        any_hit = True

        for sid in range(1, MAX_ID + 1):
            model_no, comm_result, _ = pkt.ping(ph, sid)
            if comm_result == COMM_SUCCESS:
                print(f"  id={sid:3d}  model={model_no}")
                found.append((baud, sid, model_no))
                any_hit = True
            time.sleep(0.005)
        if not any_hit:
            print("  (no replies)")

    ph.closePort()

    print(f"\nTotal motors found: {len(found)}")
    if not found:
        print("Nothing on the bus. Check arm power LED and the TTL cable into motor 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
