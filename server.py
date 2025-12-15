from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import base64
import numpy as np
from services.monitoring_service.cctv_service import camera_monitoring_service
import asyncio
import logging

app = FastAPI()

@app.websocket("/ws/{id}" )
async def websocket_endpoint(ws: WebSocket, id):
    try:
        await ws.accept()
        await camera_monitoring_service.start()
        while True:
            data = camera_monitoring_service.get_processed_frames(id)
            if data is None:
                await asyncio.sleep(0)
                continue

            await ws.send_bytes(data)

    except WebSocketDisconnect:
        await camera_monitoring_service.stop()
        logging.warning(f"Disconnected camera monitoring websocket. ID:{id}")
    except Exception:
        await camera_monitoring_service.stop()
        logging.warning(f"Failed to connect camera monitoring websocket. ID:{id}")
        await ws.close()
