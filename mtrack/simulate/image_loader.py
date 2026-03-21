"""Load images from the image folder."""

# pylint: disable=no-member

import os
import cv2
from threading import Thread, Lock, Event
from queue import Queue, Empty
from collections import OrderedDict

from ..logger import LOGGER

class LimitedSizeOrderedDict(OrderedDict):
    """An ordered dictionary with a fixed size limit that removes oldest items when full."""
    
    def __init__(self, *args, maxsize=0, **kwargs):
        self.maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.maxsize > 0:
            while len(self) > self.maxsize:
                self.popitem(last=False)

class ImageLoader:
    """Multi-threaded image loader with fixed-size cache."""
    
    def __init__(self, img_dir: str, cache_size: int = 20, num_threads: int = 8):
        """Initialize the image loader.
        
        Args:
            img_dir: Directory containing the images
            cache_size: Maximum number of images to keep in cache
            num_threads: Number of preload threads
        """
        self.img_dir = img_dir
        self.cache = LimitedSizeOrderedDict(maxsize=cache_size)
        self.cache_lock = Lock()
        
        # Loading queue and threads
        self.load_queue = Queue()
        self.threads = []
        self.running = True
        self.work_available = Event()  # Signal when work is available
        
        # Start loading threads
        for _ in range(num_threads):
            thread = Thread(target=self._load_worker, daemon=True)
            thread.start()
            self.threads.append(thread)
            
        LOGGER.info("ImageLoader initialized with %d threads and cache size %d", num_threads, cache_size)
    
    def _load_worker(self):
        """Worker thread function that continuously loads images from the queue."""
        while self.running:
            try:
                # Wait for work to be available or timeout after 5 seconds
                self.work_available.wait(timeout=5.0)
                
                try:
                    # Non-blocking get with short timeout
                    frame_num = self.load_queue.get(timeout=0.1)
                except Empty:
                    # Clear event if queue is empty
                    self.work_available.clear()
                    continue
                    
                if frame_num is None:
                    break
                    
                # Check if already in cache
                with self.cache_lock:
                    if frame_num in self.cache:
                        self.load_queue.task_done()
                        continue
                
                # Load the image
                img_path = os.path.join(self.img_dir, f"{frame_num:08d}.jpg")
                try:
                    if not os.path.exists(img_path):
                        self.load_queue.task_done()
                        continue
                    
                    img = cv2.imread(img_path)
                    if img is None:
                        LOGGER.warning("Failed to load image: %s", img_path)
                        self.load_queue.task_done()
                        continue
                        
                    # Add image to cache
                    with self.cache_lock:
                        self.cache[frame_num] = img
                        
                except Exception as e:
                    LOGGER.error("Error loading image %s: %s", img_path, str(e))
                    
                self.load_queue.task_done()
                
            except Exception as e:
                LOGGER.error("Error in load worker: %s", str(e))
                continue
    
    def preload_images(self, start_frame: int, num_frames: int = 10):
        """Preload a range of images.
        
        Args:
            start_frame: Starting frame number
            num_frames: Number of frames to preload
        """
        has_new_work = False
        for i in range(start_frame, start_frame + num_frames):
            if i not in self.cache:
                self.load_queue.put(i)
                has_new_work = True
        
        # Signal if new work was added
        if has_new_work:
            self.work_available.set()
    
    def get_image(self, frame_cnt: int) -> cv2.Mat:
        """Get image for the specified frame number.
        
        Args:
            frame_cnt: Frame number to retrieve
            
        Returns:
            cv2.Mat: Image data, or None if loading fails
        """
        # If not in cache, load this frame immediately
        if frame_cnt not in self.cache:
            img_path = os.path.join(self.img_dir, f"{frame_cnt:08d}.jpg")
            try:
                img = cv2.imread(img_path)
                if img is None:
                    LOGGER.warning("Failed to load image: %s", img_path)
                    return None
                    
                with self.cache_lock:
                    self.cache[frame_cnt] = img
            except Exception as e:
                LOGGER.error("Error loading image %s: %s", img_path, str(e))
                return None
        
        # Preload subsequent frames
        self.preload_images(frame_cnt + 1)
        
        # Return the current frame's image
        with self.cache_lock:
            return self.cache.get(frame_cnt)
    
    def clear_cache(self):
        """Clear the image cache."""
        with self.cache_lock:
            self.cache.clear()
    
    def close(self):
        """Close the loader and stop all threads."""
        self.running = False
        
        # Clear queue and add termination signals
        while not self.load_queue.empty():
            try:
                self.load_queue.get_nowait()
                self.load_queue.task_done()
            except Empty:
                break
        
        # Send termination signal to all threads
        for _ in self.threads:
            self.load_queue.put(None)
        self.work_available.set()  # Wake up all threads
        
        # Wait for all threads to finish
        for thread in self.threads:
            thread.join()
        
        # Clear the cache
        self.clear_cache()
        LOGGER.info("ImageLoader closed")
