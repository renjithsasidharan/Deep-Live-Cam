import asyncio
import websockets

async def hello():
    uri = "ws://83.60.179.66:40172/ws/video"
    async with websockets.connect(uri) as websocket:
        try:
            await websocket.send("Hello, server!")
            response = await websocket.recv()
            print(f"Received from server: {response}")
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed unexpectedly")

asyncio.get_event_loop().run_until_complete(hello())