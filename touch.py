#!/usr/bin/env python3
"""touch.py — overlay beat-synced squares and connecting lines on a video.

Usage example:
    python touch.py -i sample_data/playing_dead.mp4 -o output_with_boxes.mp4
    python touch.py (will prompt for input file interactively)
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import tempfile
from pathlib import Path
import uuid

import cv2
import librosa
import moviepy.editor as mpy
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_audio(video_path: Path, sr: int = 22050) -> Path:
    """Write the audio track of *video_path* to a temporary wav file and return its path."""
    tmp_dir = Path(tempfile.mkdtemp())
    wav_path = tmp_dir / "temp_audio.wav"
    # MoviePy uses FFmpeg under the hood
    clip = mpy.VideoFileClip(str(video_path))
    clip.audio.write_audiofile(str(wav_path), fps=sr, logger=None, verbose=False)
    return wav_path


def _detect_onsets(wav_path: Path, sr: int = 22050) -> np.ndarray:
    """Return an array of onset times (in seconds) detected in the audio file."""
    y, _ = librosa.load(str(wav_path), sr=sr)
    return librosa.onset.onset_detect(y=y, sr=sr, units="time")


class Square:
    """Data container for an on-screen square."""

    def __init__(self, born_at: float, x: int, y: int, size: int, idx: int):
        self.born_at = born_at
        self.x = x
        self.y = y
        self.size = size
        self.idx = idx

    def age(self, now: float) -> float:
        return now - self.born_at


# ---------------------------------------------------------------------------
# Tracked points pipeline
# ---------------------------------------------------------------------------

class TrackedPoint:
    """A feature point tracked across successive frames."""

    def __init__(
        self,
        pos: tuple[float, float],
        life: int,
        size: int,
        label: str,
        font_scale: float,
        text_color: tuple[int, int, int],
        vertical: bool,
    ):
        self.pos = np.array(pos, dtype=np.float32)  # shape (2,)
        self.life = life  # remaining frames
        self.size = size  # constant box size
        self.label = label  # text label shown inside box
        self.font_scale = font_scale  # scale factor for cv2.putText
        self.text_color = text_color  # BGR color tuple
        self.vertical = vertical  # whether to render text vertically


def render_tracked_effect(
    video_in: Path,
    video_out: Path,
    *,
    fps: float | None,
    pts_per_beat: int,
    ambient_rate: float,
    jitter_px: float,
    life_frames: int,
    min_size: int,
    max_size: int,
    neighbor_links: int,
    orb_fast_threshold: int,
    bell_width: float,
    seed: int | None,
):
    """Generate a video where feature points are spawned on beats and tracked with LK optical flow."""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    clip = mpy.VideoFileClip(str(video_in))
    if fps is None:
        fps = clip.fps

    # Pre-compute beat times
    wav_path = _extract_audio(video_in)
    onset_times = _detect_onsets(wav_path)
    logging.info("%d onsets detected", len(onset_times))

    # ORB detector for spawning
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=orb_fast_threshold)

    active: list[TrackedPoint] = []
    onset_idx = 0
    prev_gray: np.ndarray | None = None

    def make_frame(t: float):
        nonlocal prev_gray, onset_idx, active
        frame = clip.get_frame(t).copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        f_idx = int(round(t * fps))

        # 1. Track existing points with LK optical flow
        if prev_gray is not None and active:
            prev_pts = np.array([p.pos for p in active], dtype=np.float32).reshape(-1, 1, 2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, winSize=(21, 21), maxLevel=3
            )
            new_active: list[TrackedPoint] = []
            for tp, new_pt, ok in zip(active, next_pts.reshape(-1, 2), status.reshape(-1)):
                if not ok:
                    continue
                x, y = new_pt
                if 0 <= x < w and 0 <= y < h and tp.life > 0:
                    tp.pos = new_pt
                    tp.life -= 1
                    if jitter_px > 0:
                        tp.pos += np.random.normal(0, jitter_px, size=2)
                        tp.pos[0] = np.clip(tp.pos[0], 0, w - 1)
                        tp.pos[1] = np.clip(tp.pos[1], 0, h - 1)
                    new_active.append(tp)
            active = new_active

        # 2. Spawn new points on each beat that has passed
        while onset_idx < len(onset_times) and t >= onset_times[onset_idx]:
            logging.debug("Beat @ %.2fs — spawning points", onset_times[onset_idx])
            kps = orb.detect(gray, None)
            # Sort by response strength
            kps = sorted(kps, key=lambda k: k.response, reverse=True)
            target_spawn = random.randint(1, pts_per_beat)
            spawned = 0
            for kp in kps:
                if spawned >= target_spawn:
                    break
                x, y = kp.pt
                if any(np.linalg.norm(tp.pos - (x, y)) < 10 for tp in active):
                    continue  # too close to an existing point
                size = _sample_size_bell(min_size, max_size, bell_width)
                # generate random label
                r = random.random()
                if r < 0.33:
                    label = ''.join(random.choices('ABCDEF0123456789', k=6))
                elif r < 0.66:
                    label = str(random.randint(1, 999))
                else:
                    label = str(uuid.uuid4())[:8]

                # text styling attributes
                font_scale = random.uniform(1.0, 1.8)  # larger text than before
                text_color = random.choice([(255, 255, 255), (0, 0, 0), (255, 0, 255)])  # white/black/purple (BGR)
                vertical = random.random() < 0.25  # 25% chance to render vertically

                active.append(TrackedPoint((x, y), life_frames, size, label, font_scale, text_color, vertical))
                spawned += 1
            logging.info("Spawned %d points", spawned)
            onset_idx += 1

        # 2b. Ambient random spawns each frame (noise)
        if ambient_rate > 0:
            noise_n = np.random.poisson(ambient_rate / fps)
            for _ in range(noise_n):
                x = random.uniform(0, w)
                y = random.uniform(0, h)
                size = _sample_size_bell(min_size, max_size, bell_width)
                label_choices = [
                    ''.join(random.choices('ABCDEF0123456789', k=6)),
                    str(random.randint(1, 999)),
                    str(uuid.uuid4())[:8],
                ]
                label = random.choice(label_choices)

                # text styling attributes (ambient)
                font_scale = random.uniform(1.0, 1.8)
                text_color = random.choice([(255, 255, 255), (0, 0, 0), (255, 0, 255)])
                vertical = random.random() < 0.25

                active.append(TrackedPoint((x, y), life_frames, size, label, font_scale, text_color, vertical))

        # 3. Draw edges (nearest neighbors)
        coords = [tp.pos for tp in active]
        for i, p in enumerate(coords):
            dists = [ (j, np.linalg.norm(p - coords[j])) for j in range(len(coords)) if j != i ]
            dists.sort(key=lambda x: x[1])
            for j, _ in dists[:neighbor_links]:
                cv2.line(frame, tuple(p.astype(int)), tuple(coords[j].astype(int)), (255, 255, 255), 1)

        # 4. Draw squares for each point
        for tp in active:
            x, y = tp.pos
            s = tp.size
            tl = (int(x - s // 2), int(y - s // 2))
            br = (int(x + s // 2), int(y + s // 2))
            # Invert colors inside box for pop effect
            roi = frame[tl[1]:br[1], tl[0]:br[0]]
            if roi.size:
                frame[tl[1]:br[1], tl[0]:br[0]] = 255 - roi
            cv2.rectangle(frame, tl, br, (255, 255, 255), 1)
            # draw label text with new styling
            if tp.vertical:
                y_cursor = tl[1] + 2
                line_height = int(12 * tp.font_scale)
                for ch in tp.label:
                    cv2.putText(
                        frame,
                        ch,
                        (tl[0] + 2, y_cursor),
                        cv2.FONT_HERSHEY_PLAIN,
                        tp.font_scale,
                        tp.text_color,
                        1,
                        cv2.LINE_AA,
                    )
                    y_cursor += line_height
                    if y_cursor > br[1] - 2:
                        break  # stop if text would overflow box
            else:
                cv2.putText(
                    frame,
                    tp.label,
                    (tl[0] + 2, br[1] - 4),
                    cv2.FONT_HERSHEY_PLAIN,
                    tp.font_scale,
                    tp.text_color,
                    1,
                    cv2.LINE_AA,
                )

        prev_gray = gray
        return frame

    out_clip = mpy.VideoClip(make_frame, duration=clip.duration)
    out_clip = out_clip.set_audio(clip.audio)
    out_clip.write_videofile(str(video_out), fps=fps, codec="libx264", audio_codec="aac")


def render_intense_effect(
    video_in: Path,
    video_out: Path,
    *,
    fps: float | None,
    seed: int | None,
):
    """Generate intense effect with perfect strings from updated script."""
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    clip = mpy.VideoFileClip(str(video_in))
    if fps is None:
        fps = clip.fps

    # Pre-compute beat times
    wav_path = _extract_audio(video_in)
    onset_times = _detect_onsets(wav_path)
    logging.info("%d onsets detected", len(onset_times))

    # ORB detector for spawning
    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=15)

    # Intense preset parameters
    pts_per_beat = 40
    life_frames = 20
    jitter_px = 3.0
    neighbor_links = 4
    min_size = 10
    max_size = 35

    class IntensePoint:
        def __init__(self, pos, life, size, label, color, text_color, font_scale, vertical):
            self.pos = np.array(pos, dtype=np.float32)
            self.life = life
            self.size = size
            self.label = label
            self.color = color
            self.text_color = text_color
            self.font_scale = font_scale
            self.vertical = vertical
            self.velocity = np.random.normal(0, 2, 2).astype(np.float32)

        def update(self, dt):
            self.velocity *= 0.98  # damping
            self.pos += self.velocity * dt
            self.life -= 1

        @property
        def life_ratio(self):
            return self.life / life_frames

    # Neon color palette
    neon_colors = [
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan  
        (255, 255, 0),    # Yellow
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
    ]

    active: list[IntensePoint] = []
    onset_idx = 0
    prev_gray: np.ndarray | None = None
    trail_buffer: np.ndarray | None = None

    def make_frame(t: float):
        nonlocal prev_gray, onset_idx, active, trail_buffer
        frame = clip.get_frame(t).copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        dt = 1.0 / fps

        # Apply trail effect
        if trail_buffer is None:
            trail_buffer = np.zeros_like(frame, dtype=np.float32)
        trail_buffer = trail_buffer * 0.9 + frame.astype(np.float32) * 0.1
        frame = cv2.addWeighted(frame, 0.7, trail_buffer.astype(np.uint8), 0.3, 0)

        # Apply horizontal mirroring
        mirrored = cv2.hconcat([frame, cv2.flip(frame, 1)])
        frame = cv2.resize(mirrored, (w, h))

        # 1. Track existing points with LK optical flow
        if prev_gray is not None and active:
            prev_pts = np.array([p.pos for p in active], dtype=np.float32).reshape(-1, 1, 2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, 
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            new_active: list[IntensePoint] = []
            for point, new_pos, ok in zip(active, next_pts.reshape(-1, 2), status.reshape(-1)):
                if not ok:
                    continue
                x, y = new_pos
                if 0 <= x < w and 0 <= y < h and point.life > 0:
                    flow_vector = (new_pos - point.pos) * 1.0
                    point.velocity = point.velocity * 0.9 + flow_vector * 0.1
                    point.update(dt)
                    
                    # Apply jitter
                    if jitter_px > 0:
                        jitter = np.random.normal(0, jitter_px, 2)
                        point.pos += jitter
                    
                    # Boundary wrapping
                    point.pos[0] = point.pos[0] % w
                    point.pos[1] = point.pos[1] % h
                    
                    new_active.append(point)
            active = new_active

        # 2. Spawn new points on each beat that has passed
        while onset_idx < len(onset_times) and t >= onset_times[onset_idx]:
            logging.debug("Beat @ %.2fs — spawning points", onset_times[onset_idx])
            
            # Detect keypoints
            try:
                keypoints, descriptors = orb.detectAndCompute(gray, None)
            except:
                keypoints = orb.detect(gray, None)
            
            if not keypoints:
                keypoints = []
            
            keypoints_list = list(keypoints)
            keypoints_list.sort(key=lambda kp: kp.response, reverse=True)
            
            spawn_count = random.randint(1, pts_per_beat)
            spawned = 0
            
            for kp in keypoints_list:
                if spawned >= spawn_count:
                    break
                pos = kp.pt
                # Check minimum distance
                if any(np.linalg.norm(tp.pos - pos) < 20 for tp in active):
                    continue
                
                # Create point with intense parameters
                size = int(np.clip(np.random.normal(22.5, 4.16), min_size, max_size))
                color = random.choice(neon_colors)
                text_color = (255, 255, 255) if sum(color) < 384 else (0, 0, 0)
                
                # Label generation
                label_style = random.choice(['hex', 'number', 'uuid'])
                if label_style == 'hex':
                    label = ''.join(random.choices('0123456789ABCDEF', k=6))
                elif label_style == 'number':
                    label = str(random.randint(1, 512))
                else:
                    label = str(uuid.uuid4())[:8]
                
                font_scale = random.uniform(0.8, 1.4)
                vertical = random.random() < 0.2
                
                active.append(IntensePoint(pos, life_frames, size, label, color, text_color, font_scale, vertical))
                spawned += 1
            
            # Fill remaining with random positions
            for _ in range(spawn_count - spawned):
                x = random.uniform(0, w)
                y = random.uniform(0, h)
                size = int(np.clip(np.random.normal(22.5, 4.16), min_size, max_size))
                color = random.choice(neon_colors)
                text_color = (255, 255, 255) if sum(color) < 384 else (0, 0, 0)
                
                label_style = random.choice(['hex', 'number', 'uuid'])
                if label_style == 'hex':
                    label = ''.join(random.choices('0123456789ABCDEF', k=6))
                elif label_style == 'number':
                    label = str(random.randint(1, 512))
                else:
                    label = str(uuid.uuid4())[:8]
                
                font_scale = random.uniform(0.8, 1.4)
                vertical = random.random() < 0.2
                
                active.append(IntensePoint((x, y), life_frames, size, label, color, text_color, font_scale, vertical))
            
            logging.info("Spawned %d points", spawn_count)
            onset_idx += 1

        # 3. PERFECT STRINGS - Draw edges (nearest neighbors) from first script
        if len(active) >= 2:
            coords = [p.pos for p in active]
            for i, p in enumerate(coords):
                dists = [(j, np.linalg.norm(p - coords[j])) 
                        for j in range(len(coords)) if j != i]
                dists.sort(key=lambda x: x[1])
                for j, _ in dists[:neighbor_links]:
                    # Always white, no fading - PERFECT STRINGS
                    cv2.line(frame, 
                            tuple(p.astype(int)), 
                            tuple(coords[j].astype(int)), 
                            (255, 255, 255),  # Pure white
                            1)

        # 4. Draw squares for each point
        for point in active:
            x, y = point.pos.astype(int)
            s = point.size
            
            # Calculate bounds with clipping
            x1, y1 = max(0, x - s//2), max(0, y - s//2)
            x2, y2 = min(frame.shape[1], x + s//2), min(frame.shape[0], y + s//2)
            
            if x1 >= x2 or y1 >= y2:
                continue
                
            # Invert interior
            roi = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = 255 - roi
            
            # Draw border
            cv2.rectangle(frame, (x1, y1), (x2, y2), point.color, 1)
            
            # Draw label
            if point.vertical:
                y_cursor = y1 + 2
                line_height = int(12 * point.font_scale)
                for ch in point.label:
                    cv2.putText(frame, ch, (x1 + 2, y_cursor),
                               cv2.FONT_HERSHEY_PLAIN, point.font_scale, point.text_color, 1)
                    y_cursor += line_height
                    if y_cursor > y2 - 2:
                        break
            else:
                cv2.putText(frame, point.label, (x1 + 2, y2 - 4),
                           cv2.FONT_HERSHEY_PLAIN, point.font_scale, point.text_color, 1)

        # Apply glow effect
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=3, sigmaY=3)
        frame = cv2.addWeighted(frame, 1.0, blurred, 0.8, 0)

        prev_gray = gray.copy()
        return frame

    out_clip = mpy.VideoClip(make_frame, duration=clip.duration)
    out_clip = out_clip.set_audio(clip.audio)
    out_clip.write_videofile(str(video_out), fps=fps, codec="libx264", audio_codec="aac")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _sample_size_bell(min_s: int, max_s: int, width_div: float = 6.0) -> int:
    """Sample size with adjustable bell width; smaller width_div => wider distribution."""
    mean = (min_s + max_s) / 2.0
    sigma = (max_s - min_s) / width_div
    for _ in range(10):
        val = np.random.normal(mean, sigma)
        if min_s <= val <= max_s:
            return int(val)
    # fallback clamp
    return int(np.clip(val, min_s, max_s))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _get_valid_video_path() -> Path:
    """Interactively prompt user for a valid video file path."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    while True:
        user_input = input("Enter the path to your input video file: ").strip()
        
        # Remove quotes that might be added by drag-and-drop
        user_input = user_input.strip('"\'')
        
        path = Path(user_input)
        
        if not path.exists():
            print(f"Error: File '{path}' does not exist. Please try again.")
            continue
            
        if path.suffix.lower() not in video_extensions:
            print(f"Error: '{path.suffix}' is not a recognized video format. Supported formats: {', '.join(video_extensions)}")
            continue
            
        return path


def _select_effect() -> int:
    """Prompt user to select which effect to use."""
    print("\n" + "="*50)
    print("SELECT VISUAL EFFECT")
    print("="*50)
    print("1. ORIGINAL EFFECT")
    print("   - Clean squares with white connections")
    print("   - Simple and elegant")
    print()
    print("2. INTENSE EFFECT") 
    print("   - High-density neon squares with perfect strings")
    print("   - Motion trails, mirroring, and glow effects")
    print("   - Chaotic and energetic")
    print("="*50)
    
    while True:
        try:
            choice = int(input("\nChoose effect (1 or 2): ").strip())
            if choice in [1, 2]:
                return choice
            else:
                print("Please enter 1 or 2")
        except ValueError:
            print("Please enter a valid number (1 or 2)")


def _parse_args():
    p = argparse.ArgumentParser(description="Overlay beat-synced boxes + lines on a video.")
    p.add_argument("-i", "--input", type=Path, help="Input video file")
    p.add_argument("-o", "--output", type=Path, help="Output video file")
    p.add_argument("--fps", type=float, default=None, help="FPS for output (default: same as source)")
    p.add_argument("--life-frames", type=int, default=10, help="How many frames a point remains alive (short for crazy mode)")
    p.add_argument("--pts-per-beat", type=int, default=20, help="Maximum new points to spawn on each beat (actual number is random up to this)")
    p.add_argument("--ambient-rate", type=float, default=5.0, help="Average number of random points spawned per second even in silence")
    p.add_argument("--jitter-px", type=float, default=0.5, help="Per-frame positional jitter in pixels for organic motion")
    p.add_argument("--min-size", type=int, default=15, help="Minimum square size in pixels")
    p.add_argument("--max-size", type=int, default=40, help="Maximum square size in pixels")
    p.add_argument("--neighbor-links", type=int, default=3, help="Number of neighbor edges per point")
    p.add_argument("--orb-fast-threshold", type=int, default=20, help="FAST threshold for ORB detector (lower = more keypoints)")
    p.add_argument("--bell-width", type=float, default=4.0, help="Divisor controlling bell curve width (smaller -> wider)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    
    args = p.parse_args()
    
    # Interactive input handling
    if args.input is None:
        args.input = _get_valid_video_path()
    
    # Effect selection
    effect_choice = _select_effect()
    
    if args.output is None:
        # Auto-generate output path based on effect choice
        if effect_choice == 1:
            output_filename = f"{args.input.stem}_1_effect.mp4"
        else:
            output_filename = f"{args.input.stem}_2_effect.mp4"
        args.output = args.input.parent / output_filename
        print(f"Output file will be: {args.output}")
    
    args.effect_choice = effect_choice
    return args


def main():
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s:%(message)s")

    print(f"\nProcessing video with {'ORIGINAL' if args.effect_choice == 1 else 'INTENSE'} effect...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    if args.effect_choice == 1:
        # Original effect
        render_tracked_effect(
            video_in=args.input,
            video_out=args.output,
            fps=args.fps,
            pts_per_beat=args.pts_per_beat,
            ambient_rate=args.ambient_rate,
            jitter_px=args.jitter_px,
            life_frames=args.life_frames,
            min_size=args.min_size,
            max_size=args.max_size,
            neighbor_links=args.neighbor_links,
            orb_fast_threshold=args.orb_fast_threshold,
            bell_width=args.bell_width,
            seed=args.seed,
        )
    else:
        # Intense effect with perfect strings
        render_intense_effect(
            video_in=args.input,
            video_out=args.output,
            fps=args.fps,
            seed=args.seed,
        )
    
    print(f"Processing complete! Output saved to: {args.output}")


if __name__ == "__main__":
    main()
