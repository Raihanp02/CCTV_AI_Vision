from fastapi import WebSocket, WebSocketDisconnect
import cv2
import asyncio

@router.websocket("/ws/vision")
async def vision_ws(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            async with shared_state.lock:
                frame = shared_state.frame
                info = shared_state.info

            if frame is None:
                await asyncio.sleep(0.01)
                continue

            # Encode frame
            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            await websocket.send_bytes(jpeg.tobytes())

    except WebSocketDisconnect:
        print("WebSocket disconnected")
