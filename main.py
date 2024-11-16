import sys
import signal
import sounddevice as sd
import numpy as np
import tkinter as tk
import queue
from contextlib import contextmanager

# 設定
SAMPLE_RATE = 44100
WINDOW_SIZE = 1024
NUM_BARS = 60
NOISE_THRESHOLD = 0.05


@contextmanager
def signal_handler():
    """シグナルハンドラのコンテキストマネージャ"""
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)


class SpectrumVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Spectrum Visualizer")
        self.is_running = True

        # キャンバスの設定
        self.canvas = tk.Canvas(root, width=750, height=400, bg="black")
        self.canvas.pack()

        # バーの初期化
        self.bars = []
        self._initialize_bars()

    def _initialize_bars(self):
        bar_width = 800 // NUM_BARS
        for i in range(NUM_BARS):
            x0 = i * bar_width
            x1 = x0 + bar_width - 2
            bar = self.canvas.create_rectangle(x0, 400, x1, 400, fill="purple")
            self.bars.append(bar)

    def update_bars(self, spectrum):
        if not self.is_running:
            return

        max_height = 400
        bar_width = 800 // NUM_BARS

        if np.all(spectrum == 0):
            for bar in self.bars:
                try:
                    self.canvas.coords(bar, 0, 0, 0, 0)
                except tk.TclError:
                    return
            return

        try:
            spectrum = np.interp(spectrum, (0, max(spectrum)), (0, max_height))
            spectrum = np.clip(spectrum, 0, max_height)

            for i, bar in enumerate(self.bars):
                height = max_height - spectrum[i]
                x0 = i * bar_width
                x1 = (i + 1) * bar_width - 2
                self.canvas.coords(bar, x0, height, x1, max_height)
        except tk.TclError:
            self.is_running = False
            return

    def stop(self):
        self.is_running = False


class AudioStream:
    def __init__(self, data_queue):
        self.data_queue = data_queue
        self.stream = None
        self.is_running = True

    def start(self):
        self.stream = sd.InputStream(
            channels=2,
            samplerate=SAMPLE_RATE,
            callback=self._audio_callback
        )
        self.stream.start()

    def _audio_callback(self, indata, frames, time, status):
        if not self.is_running:
            return

        try:
            fft_data = np.abs(np.fft.rfft(indata[:, 0], n=WINDOW_SIZE))
            fft_data = fft_data[:NUM_BARS]
            fft_data = np.where(fft_data > NOISE_THRESHOLD, fft_data, 0)
            self.data_queue.put_nowait(fft_data)
        except queue.Full:
            pass

    def stop(self):
        self.is_running = False
        if self.stream is not None:
            try:
                if self.stream.active:
                    self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error closing stream: {e}")


class Application:
    def __init__(self):
        self.root = tk.Tk()
        self.data_queue = queue.Queue(maxsize=100)  # キューサイズを制限
        self.visualizer = SpectrumVisualizer(self.root)
        self.audio_stream = AudioStream(self.data_queue)
        self.is_running = True

        # 終了処理のセットアップ
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        print("stopping application...")
        self.stop()

    def process_queue(self):
        if not self.is_running:
            return

        try:
            while not self.data_queue.empty() and self.is_running:
                spectrum = self.data_queue.get_nowait()
                self.visualizer.update_bars(spectrum)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Queue processing error: {e}")

        if self.is_running:
            self.root.after(16, self.process_queue)

    def stop(self):
        """アプリケーションの停止処理"""
        self.is_running = False
        self.visualizer.stop()
        self.audio_stream.stop()

        try:
            self.root.after_cancel(self.root.after(16, self.process_queue))
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

        print("Application stopped.")

    def run(self):
        """アプリケーションの実行"""
        sd.default.device = [3, 5]
        self.audio_stream.start()
        self.root.after(16, self.process_queue)
        self.root.mainloop()


def main():
    try:
        Application().run()
    except Exception as e:
        print(f"Main error: {e}")
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()