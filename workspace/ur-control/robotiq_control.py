#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robotiq 2F gripper control and readout via URCap ASCII socket (default port 63352).

This client talks to the Universal Robots controller IP where the Robotiq URCap
runs and proxies the gripper over an ASCII protocol. Typical commands:
  - SET ACT 1        # activate
  - SET GTO 1        # go to requested position
  - SET POS <0-255>  # requested position (0=open, 255=close for 2F)
  - SET SPE <0-255>  # speed
  - SET FOR <0-255>  # force
  - GET POS|SPE|FOR|STA|OBJ  # query current state

Assumptions:
  - Robotiq 2F gripper connected to UR robot with Robotiq URCap installed
  - UR controller is reachable over the network (e.g., 192.168.0.2)
  - URCap ASCII server is enabled at TCP port 63352

Notes:
  - This module provides basic activate/open/close/move and status readout.
  - Position is returned in raw [0, 255]. You may convert to width (mm) if you
	know your gripper's max width (e.g., 85 or 140).
  - On some setups, open/close might be inverted. If so, set invert=True.

Usage examples:
  python workspace/ur-study/robotiq_control.py --ip 192.168.0.2 activate
  python workspace/ur-study/robotiq_control.py --ip 192.168.0.2 open
  python workspace/ur-study/robotiq_control.py --ip 192.168.0.2 close
  python workspace/ur-study/robotiq_control.py --ip 192.168.0.2 move --pos 128 --speed 200 --force 150
  python workspace/ur-study/robotiq_control.py --ip 192.168.0.2 status

References:
  - Robotiq C-Model (2F) registers and URCap ASCII control conventions
"""

from __future__ import annotations

import socket
import time
import argparse
from typing import Dict, Optional


DEFAULT_PORT = 63352  # Robotiq URCap ASCII server on UR controller
RECV_BUF = 1024


class RobotiqURCapClient:
	"""Simple Robotiq 2F gripper client through URCap ASCII port.

	Contract:
	  - Inputs: ip (str), port (int)
	  - Operations: activate(), open(), close(), move(pos), set_speed(), set_force(), get_status()
	  - Outputs: status dict with raw registers; position_raw [0..255]
	  - Error modes: socket timeout/connection errors -> raises OSError
	"""

	def __init__(self, ip: str, port: int = DEFAULT_PORT, timeout: float = 1.0, invert: bool = False):
		self.ip = ip
		self.port = port
		self.timeout = timeout
		self.invert = invert
		self._sock: Optional[socket.socket] = None

	# ------------- connection -------------
	def connect(self):
		if self._sock is not None:
			return
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.settimeout(self.timeout)
		s.connect((self.ip, self.port))
		self._sock = s

	def close(self):
		if self._sock is not None:
			try:
				self._sock.close()
			finally:
				self._sock = None

	def __enter__(self):
		self.connect()
		return self

	def __exit__(self, exc_type, exc, tb):
		self.close()

	# ------------- low-level I/O -------------
	def _send(self, msg: str) -> None:
		if self._sock is None:
			raise RuntimeError("Not connected")
		data = (msg.strip() + "\r\n").encode("ascii")
		self._sock.sendall(data)

	def _recv_line(self) -> str:
		if self._sock is None:
			raise RuntimeError("Not connected")
		# Many URCap builds reply with a single line per request
		data = self._sock.recv(RECV_BUF)
		if not data:
			return ""
		return data.decode("ascii", errors="ignore").strip()

	def _cmd(self, msg: str) -> str:
		self._send(msg)
		return self._recv_line()

	# ------------- helpers -------------
	@staticmethod
	def _clip_byte(x: int) -> int:
		return int(max(0, min(255, int(x))))

	@staticmethod
	def _parse_kv(resp: str) -> Dict[str, int]:
		# Expected forms like "POS 123" or space-separated pairs
		out: Dict[str, int] = {}
		parts = resp.split()
		if len(parts) >= 2:
			# Assume first is key, last is integer value
			key = parts[0].upper()
			try:
				val = int(parts[-1])
			except ValueError:
				# Fallback: no integer convertible value
				return {}
			out[key] = val
		return out

	# ------------- high-level API -------------
	def activate(self, wait: bool = True, timeout_s: float = 5.0) -> None:
		"""Activate gripper; optionally wait until status is ready.

		Typical sequence: SET ACT 1; SET GTO 1
		"""
		self._cmd("SET ACT 1")
		time.sleep(0.05)
		self._cmd("SET GTO 1")
		if wait:
			t0 = time.time()
			while time.time() - t0 < timeout_s:
				sta = self.get_status().get("STA", None)
				# Heuristic: STA 3 -> activated and ready
				if sta == 3:
					return
				time.sleep(0.1)
			# Timeout; continue without raising to be tolerant

	def set_speed(self, speed: int) -> None:
		speed = self._clip_byte(speed)
		self._cmd(f"SET SPE {speed}")

	def set_force(self, force: int) -> None:
		force = self._clip_byte(force)
		self._cmd(f"SET FOR {force}")

	def move(self, pos: int, speed: Optional[int] = None, force: Optional[int] = None) -> None:
		"""Move to raw position [0..255]. Optionally set speed/force first."""
		if speed is not None:
			self.set_speed(speed)
		if force is not None:
			self.set_force(force)
		p = self._clip_byte(pos)
		if self.invert:
			p = 255 - p
		self._cmd(f"SET POS {p}")
		time.sleep(0.02)
		self._cmd("SET GTO 1")

	def open(self, speed: Optional[int] = None, force: Optional[int] = None) -> None:
		self.move(0, speed=speed, force=force)

	def close(self, speed: Optional[int] = None, force: Optional[int] = None) -> None:
		self.move(255.0, speed=speed, force=force)

	def get_status(self) -> Dict[str, int]:
		"""Read a snapshot of gripper state.

		Returns keys (when available from URCap):
		  - POS: current position [0..255]
		  - SPE: speed [0..255]
		  - FOR: force [0..255]
		  - STA: status code
		  - OBJ: object detection code
		"""
		result: Dict[str, int] = {}
		for key in ("POS", "SPE", "FOR", "STA", "OBJ"):
			resp = self._cmd(f"GET {key}")
			kv = self._parse_kv(resp)
			result.update(kv)
		# If inverted, reflect reported position back to normal frame
		if self.invert and "POS" in result:
			result["POS"] = 255 - result["POS"]
		return result

	@staticmethod
	def pos_to_width_mm(pos: int, max_width_mm: float = 85.0, invert: bool = False) -> float:
		"""Convert raw position [0..255] to approximate width in mm.

		This assumes a linear mapping: 0 -> open (max_width), 255 -> closed (0 mm).
		Set invert=True if your wiring reports the reverse.
		"""
		pos_clamped = max(0, min(255, int(pos)))
		if invert:
			pos_clamped = 255 - pos_clamped
		# 0 -> max_width, 255 -> 0
		return max_width_mm * (255 - pos_clamped) / 255.0


def main():
	parser = argparse.ArgumentParser(description="Control Robotiq 2F via URCap ASCII server")
	parser.add_argument("--action", default="open", choices=["activate", "open", "close", "move", "status"], help="Action to perform")
	parser.add_argument("--ip", default="192.168.12.21", help="UR controller IP hosting Robotiq URCap")
	parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="URCap ASCII port (default 63352)")
	parser.add_argument("--pos", type=int, default=None, help="Target position [0..255] for move action")
	parser.add_argument("--speed", type=int, default=200, help="Speed [0..255]")
	parser.add_argument("--force", type=int, default=150, help="Force [0..255]")
	parser.add_argument("--invert", action="store_true", help="Invert position direction (if wiring inverted)")
	parser.add_argument("--max-width-mm", type=float, default=None, help="If provided, also print width in mm when reading status")

	args = parser.parse_args()

	with RobotiqURCapClient(args.ip, port=args.port, invert=args.invert) as g:
		if args.action == "activate":
			g.activate(wait=True)
			print("Activated.")
		elif args.action == "open":
			g.open(speed=args.speed, force=args.force)
			print("Open command sent.")
		elif args.action == "close":
			g.close(speed=args.speed, force=args.force)
			print("Close command sent.")
		elif args.action == "move":
			if args.pos is None:
				raise SystemExit("--pos is required for move action")
			g.move(args.pos, speed=args.speed, force=args.force)
			print(f"Move command sent to pos={args.pos}.")
		elif args.action == "status":
			st = g.get_status()
			if st:
				if args.max_width_mm is not None and "POS" in st:
					width = RobotiqURCapClient.pos_to_width_mm(st["POS"], max_width_mm=args.max_width_mm, invert=args.invert)
					st["WIDTH_MM"] = round(width, 3)
				print(st)
			else:
				print("No status received (check URCap and connection)")


if __name__ == "__main__":
	main()

