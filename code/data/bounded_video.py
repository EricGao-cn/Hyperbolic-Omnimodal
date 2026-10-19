"""Isolate native video reads so a stalled decoder cannot block a data worker."""
import io
import json
import os
import random
from pathlib import Path
import select
import struct
import subprocess
import sys
import time

import numpy as np


class BoundedVideoReader:
    def __init__(self, timeout=10, trace_dir=None):
        self.timeout = timeout
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.process = None

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()

    def _start(self):
        if self.process is None or self.process.poll() is not None:
            self.close()
            self.process = subprocess.Popen(
                [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--worker'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=0,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'),
            )

    def _read(self, length, deadline):
        chunks = []
        descriptor = self.process.stdout.fileno()
        while length:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
                raise TimeoutError('isolated Decord read timed out')
            chunk = os.read(descriptor, min(length, 1024 * 1024))
            if not chunk:
                raise RuntimeError('isolated Decord process closed its output')
            chunks.append(chunk)
            length -= len(chunk)
        return b''.join(chunks)

    def request(self, command):
        self._start()
        self.process.stdin.write((json.dumps(command) + '\n').encode())
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout
        size = struct.unpack('!I', self._read(4, deadline))[0]
        header = json.loads(self._read(size, deadline))
        if 'error' in header:
            raise RuntimeError(header['error'])
        if 'shape' in header:
            data = self._read(int(np.prod(header['shape'])), deadline)
            return np.frombuffer(data, dtype=np.uint8).reshape(header['shape']).copy()
        return header

    def _record(self, path, phase, error=None, **details):
        if self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            record = {'path': str(path), 'phase': phase, 'pid': os.getpid(),
                      'rank': os.environ.get('RANK'), 'time': time.time()}
            record.update(details)
            if error is not None:
                record['error'] = str(error)
            marker = self.trace_dir / f'worker_{os.getpid()}.json'
            marker.write_text(json.dumps(record))
            if error is not None:
                with (self.trace_dir / f'fallbacks_{os.getpid()}.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')

    def read(self, path, choose_indices):
        self._record(path, 'decord_metadata')
        try:
            metadata = self.request({'open': str(path)})
        except Exception as error:
            self.close()
            self._record(path, 'ffmpeg_metadata', error)
            count = ffmpeg_frame_count(path)
            indices = choose_indices(count, None)
            frames = ffmpeg_frames(path, indices)
        else:
            sample_rng = random.getstate()
            indices = choose_indices(metadata['frames'], metadata['fps'])
            self._record(path, 'decord_frames')
            try:
                frames = self.request({'indices': indices})
            except Exception as error:
                self.close()
                self._record(path, 'ffmpeg_frames', error)
                count = ffmpeg_frame_count(path)
                if count != metadata['frames']:
                    # Damaged containers can advertise frames that cannot be decoded.
                    # Reapply the same random sampling policy to the real frame count.
                    random.setstate(sample_rng)
                    indices = choose_indices(count, None)
                    self._record(path, 'ffmpeg_frame_count_difference', error,
                                 container_frames=metadata['frames'], decoded_frames=count,
                                 selected_indices=indices)
                frames = ffmpeg_frames(path, indices)
        self._record(path, 'complete')
        return frames

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def ffmpeg_command():
    import imageio_ffmpeg
    return [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error', '-threads', '1']


def ffmpeg_frame_count(path):
    result = subprocess.run(ffmpeg_command() + ['-i', str(path), '-map', '0:v:0',
        '-an', '-vsync', '0', '-progress', 'pipe:1', '-f', 'null', '-'],
        capture_output=True, check=True, timeout=120)
    counts = [int(line.split('=', 1)[1]) for line in result.stdout.decode().splitlines()
              if line.startswith('frame=')]
    if not counts or counts[-1] <= 0:
        raise RuntimeError(f'No video frames decoded: {path}')
    return counts[-1]


def ffmpeg_frames(path, indices):
    from PIL import Image
    unique = sorted(set(indices))
    expression = '+'.join(f'eq(n\\,{index})' for index in unique)
    result = subprocess.run(ffmpeg_command() + ['-i', str(path), '-map', '0:v:0',
        '-an', '-vf', 'select=' + expression, '-vsync', '0', '-threads', '1',
        '-c:v', 'png', '-f', 'image2pipe', 'pipe:1'],
        capture_output=True, check=True, timeout=120)
    data, offset, frames = result.stdout, 0, []
    while offset < len(data):
        start = offset
        if data[offset:offset+8] != b'\x89PNG\r\n\x1a\n':
            raise RuntimeError(f'Invalid PNG stream decoded from {path}')
        offset += 8
        while True:
            size = struct.unpack('!I', data[offset:offset+4])[0]
            kind = data[offset+4:offset+8]
            offset += size + 12
            if kind == b'IEND':
                break
        frames.append(np.asarray(Image.open(io.BytesIO(data[start:offset])).convert('RGB')))
    if len(frames) != len(unique):
        raise RuntimeError(f'{path}: expected {len(unique)} selected frames, got {len(frames)}')
    lookup = dict(zip(unique, frames))
    return np.stack([lookup[index] for index in indices])


def worker():
    import decord
    reader = None
    for line in sys.stdin:
        payload = None
        try:
            command = json.loads(line)
            if 'open' in command:
                reader = decord.VideoReader(command['open'], num_threads=1)
                header = {'frames': len(reader), 'fps': float(reader.get_avg_fps())}
            else:
                frames = reader.get_batch(command['indices']).asnumpy()
                header = {'shape': frames.shape}
                payload = frames.tobytes()
        except Exception as error:
            header = {'error': repr(error)}
        encoded = json.dumps(header).encode()
        sys.stdout.buffer.write(struct.pack('!I', len(encoded)) + encoded)
        if payload is not None:
            sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()


if __name__ == '__main__':
    worker()
