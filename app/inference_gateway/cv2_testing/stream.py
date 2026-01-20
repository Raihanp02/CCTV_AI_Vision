import cv2

class StreamVideo:
    def __init__(self, worker, fps=30):
        self.worker = worker
        self.fps = fps

    def start(self):
        while True:
            frame_info = self.worker.vision_buffer.get()
            for cam_id, info in frame_info.items():
                for frame in info["frame"]:
                    cv2.imshow(f"Camera: {cam_id}", frame)
                    if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                        self.worker.stop()
                        cv2.destroyAllWindows()
                        return