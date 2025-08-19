import time
import sys

class SimpleProgressBar:
    """
    A simple progress bar that can be used in distributed training.
    Only prints on the main process.
    """
    def __init__(self, total, desc=""):
        self.total = total
        self.current = 0
        self.start_time = time.time()
        self.desc = desc

    def _format_time(self, seconds):
        """Format seconds to HH:MM:SS"""
        if seconds == float('inf') or seconds < 0:
            return "??:??:??"
        return time.strftime('%H:%M:%S', time.gmtime(seconds))

    def update(self, loss = None, n=1):
        """Update the progress bar after each iteration"""
        self.current += n

        # Only print on the main process
        elapsed_time = time.time() - self.start_time
        
        # Calculate progress
        progress = self.current / self.total
        
        # Calculate speed
        speed = self.current / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate estimated remaining time (ETA)
        remaining_items = self.total - self.current
        eta_seconds = remaining_items / speed if speed > 0 else float('inf')
        
        # Format time
        elapsed_str = self._format_time(elapsed_time)
        eta_str = self._format_time(eta_seconds)
        
        # Build the progress bar
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Combine all information
        display_str = (
            f"{self.desc} |{bar}| {self.current}/{self.total} [{progress:.1%}] | "
            f"Elapsed: {elapsed_str} | ETA: {eta_str} | {speed:.2f} it/s"
        )

        if loss is not None:
            display_str += f" | Loss: {loss:.4f}"

        print(f"\r{display_str} ", end="", flush=True)

    def close(self):
        """Call after the loop ends, print a newline to avoid the next line overwriting the progress bar"""
        # Print a newline to complete the progress bar display
        print()
