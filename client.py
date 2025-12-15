import asyncio
import websockets

async def test_fastapi():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        print(await ws.recv())
        await ws.send("ping")
        print(await ws.recv())

asyncio.run(test_fastapi())
